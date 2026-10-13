# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Deterministic FP32/BF16 audio comparison for the CUDA runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

import numpy as np


def _signal_stats(samples: np.ndarray, sample_rate: int = 48000) -> dict:
  signal = samples.astype(np.float64) / 32768.0
  mono = signal.mean(axis=1)
  window_size = 4096
  usable = len(mono) // window_size * window_size
  windows = mono[:usable].reshape(-1, window_size)
  spectrum = np.abs(np.fft.rfft(windows, axis=1)) ** 2
  frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
  power = spectrum.sum(axis=1)
  centroid = (
      (spectrum * frequencies).sum(axis=1) / np.maximum(power, 1e-20)
  )
  high_frequency = spectrum[:, frequencies >= 12000].sum(axis=1)
  left = signal[:, 0] - signal[:, 0].mean()
  right = signal[:, 1] - signal[:, 1].mean()
  correlation_denominator = np.sqrt(np.sum(left**2) * np.sum(right**2))
  stereo_correlation = (
      float(np.sum(left * right) / correlation_denominator)
      if correlation_denominator > 0
      else 0.0
  )
  return {
      "rms": float(np.sqrt(np.mean(signal**2))),
      "peak": float(np.abs(signal).max()),
      "clipped_fraction": float(np.mean(np.abs(samples) >= 32767)),
      "dc_offset": [float(x) for x in signal.mean(axis=0)],
      "stereo_correlation": stereo_correlation,
      "spectral_centroid_hz": float(np.mean(centroid)),
      "high_frequency_power_fraction": float(
          np.mean(high_frequency / np.maximum(power, 1e-20))
      ),
      "finite": bool(np.isfinite(signal).all()),
  }


def _write_wav(path: Path, samples: np.ndarray) -> None:
  with wave.open(str(path), "wb") as handle:
    handle.setnchannels(2)
    handle.setsampwidth(2)
    handle.setframerate(48000)
    handle.writeframes(samples.astype("<i2", copy=False).tobytes())


def _read_wav(path: Path) -> np.ndarray:
  with wave.open(str(path), "rb") as handle:
    assert handle.getnchannels() == 2
    return np.frombuffer(handle.readframes(handle.getnframes()), "<i2").reshape(-1, 2)


def _capture(asset_root: str, model_name: str, physical_device: int,
             frames: int, parameter_dtype: str, codec_compute_dtype: str,
             output: str) -> dict:
  os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
  os.environ["MAGENTA_RT_HOME"] = asset_root
  os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path(asset_root) / "jax-cache")
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

  from magenta_rt import MagentaRT2Jax, paths  # pylint: disable=import-outside-toplevel
  from magenta_rt.config import MUSICCOCA  # pylint: disable=import-outside-toplevel

  paths.set_magenta_home(asset_root)
  model = MagentaRT2Jax(
      size=model_name,
      parameter_dtype=parameter_dtype,
      codec_compute_dtype=codec_compute_dtype,
  )
  embedding = model.embed_style("disco funk", use_mapper=True)
  prepared = model.prepare_conditioning({MUSICCOCA.key: embedding})
  state = model.initial_state()
  chunks = []
  for _ in range(frames):
    pcm, state = model.stream_step(prepared, state)
    chunks.append(pcm)
  samples = np.concatenate(chunks)
  target = Path(output)
  _write_wav(target, samples)
  return {"path": str(target), "frames": frames, **_signal_stats(samples)}


def compare_precision(asset_root: str, model: str, device: int,
                      frames: int = 250) -> dict:
  output_dir = Path(asset_root) / "quality"
  output_dir.mkdir(parents=True, exist_ok=True)
  records = {}
  configurations = {
      "fp32": ("float32", "float32"),
      "bf16": ("bfloat16", "bfloat16"),
  }
  for label, (parameter_dtype, codec_compute_dtype) in configurations.items():
    output = output_dir / f"{model}-{label}.wav"
    command = [
        sys.executable, "-m", "magenta_rt.web.quality", "--worker",
        "--asset-root", str(Path(asset_root).resolve()),
        "--model", model, "--device", str(device), "--frames", str(frames),
        "--parameter-dtype", parameter_dtype,
        "--codec-compute-dtype", codec_compute_dtype,
        "--output", str(output),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
      raise RuntimeError(process.stderr.strip() or f"{label} capture failed")
    records[label] = json.loads(process.stdout.strip().splitlines()[-1])

  fp32 = _read_wav(Path(records["fp32"]["path"])).astype(np.float64)
  bf16 = _read_wav(Path(records["bf16"]["path"])).astype(np.float64)
  error = fp32 - bf16
  signal_power = np.mean(fp32**2)
  error_power = np.mean(error**2)
  report = {
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "model": model,
      "physical_device": device,
      "prompt": "disco funk",
      "records": records,
      "sample_correlation": float(np.corrcoef(fp32.ravel(), bf16.ravel())[0, 1]),
      "difference_snr_db": float(
          10 * np.log10(signal_power / max(error_power, 1e-20))
      ),
  }
  target = output_dir / f"{model}-precision-report.json"
  target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(f"Saved {target}")
  return report


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--asset-root", required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--device", required=True, type=int)
  parser.add_argument("--frames", type=int, default=250)
  parser.add_argument("--parameter-dtype", default="float32")
  parser.add_argument("--codec-compute-dtype", default="float32")
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  if not args.worker:
    parser.error("this module entry point is reserved for capture workers")
  result = _capture(
      args.asset_root, args.model, args.device, args.frames,
      args.parameter_dtype, args.codec_compute_dtype, args.output,
  )
  print(json.dumps(result))


if __name__ == "__main__":
  main()
