# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production-equivalent CUDA latency benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from magenta_rt.web.manager import MODEL_PRECISION


def _gpu_memory_mib(device: int) -> int | None:
  result = subprocess.run(
      [
          "nvidia-smi",
          f"--id={device}",
          "--query-gpu=memory.used",
          "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  try:
    return int(result.stdout.strip())
  except ValueError:
    return None


def _worker(asset_root: str, model_name: str, physical_device: int,
            warmup: int, steps: int, parameter_dtype: str = "float32",
            codec_compute_dtype: str = "float32") -> dict:
  os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
  os.environ["MAGENTA_RT_HOME"] = asset_root
  os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(asset_root) / "jax-cache"))
  os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
  os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

  import jax  # pylint: disable=import-outside-toplevel
  import numpy as np  # pylint: disable=import-outside-toplevel

  from magenta_rt import MagentaRT2Jax  # pylint: disable=import-outside-toplevel
  from magenta_rt.config import MUSICCOCA  # pylint: disable=import-outside-toplevel
  from magenta_rt import paths  # pylint: disable=import-outside-toplevel

  paths.set_magenta_home(asset_root)
  cache_dir = Path(asset_root) / "jax-cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  try:
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  except AttributeError:
    pass

  started = time.perf_counter()
  model = MagentaRT2Jax(
      size=model_name,
      parameter_dtype=parameter_dtype,
      codec_compute_dtype=codec_compute_dtype,
  )
  embedding = model.embed_style("disco funk", use_mapper=True)
  prepared = model.prepare_conditioning({MUSICCOCA.key: embedding})
  state = model.initial_state()
  startup_seconds = time.perf_counter() - started

  for _ in range(warmup):
    _, state = model.stream_step(prepared, state)

  latencies = []
  peak = _gpu_memory_mib(physical_device) or 0
  last_pcm = None
  for index in range(steps):
    tick = time.perf_counter_ns()
    last_pcm, state = model.stream_step(prepared, state)
    latencies.append((time.perf_counter_ns() - tick) / 1_000_000)
    if index % 25 == 0:
      peak = max(peak, _gpu_memory_mib(physical_device) or 0)

  values = np.asarray(latencies, dtype=np.float64)
  assert last_pcm is not None
  return {
      "model": model_name,
      "physical_device": physical_device,
      "parameter_dtype": parameter_dtype,
      "codec_compute_dtype": codec_compute_dtype,
      "jax_device": str(jax.devices()[0]),
      "startup_seconds": startup_seconds,
      "warmup_frames": warmup,
      "measured_frames": steps,
      "mean_ms": float(statistics.fmean(latencies)),
      "p50_ms": float(np.percentile(values, 50)),
      "p95_ms": float(np.percentile(values, 95)),
      "p99_ms": float(np.percentile(values, 99)),
      "max_ms": float(values.max()),
      "steps_per_second": 1000.0 / float(statistics.fmean(latencies)),
      "peak_gpu_memory_mib": peak,
      "pcm_peak": int(np.abs(last_pcm.astype(np.int32)).max()),
      "realtime": bool(np.percentile(values, 95) <= 40.0),
  }


def run_matrix(asset_root: str, models: list[str], devices: list[int],
               warmup: int = 50, steps: int = 1000) -> dict:
  """Run every model/device pair in an isolated subprocess."""
  root = Path(asset_root).resolve()
  output_dir = root / "benchmarks"
  output_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for device in devices:
    for model in models:
      print(f"Benchmarking {model} on physical GPU {device}...", flush=True)
      command = [
          sys.executable,
          "-m",
          "magenta_rt.web.benchmark",
          "--worker",
          "--asset-root",
          str(root),
          "--model",
          model,
          "--device",
          str(device),
          "--warmup",
          str(warmup),
          "--steps",
          str(steps),
      ]
      parameter_dtype, codec_compute_dtype = MODEL_PRECISION[model]
      command.extend([
          "--parameter-dtype", parameter_dtype,
          "--codec-compute-dtype", codec_compute_dtype,
      ])
      proc = subprocess.run(command, capture_output=True, text=True, check=False)
      if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        results.append({
            "model": model,
            "physical_device": device,
            "error": proc.stderr.strip() or f"worker exited {proc.returncode}",
        })
        continue
      record = json.loads(proc.stdout.strip().splitlines()[-1])
      results.append(record)
      print(
          f"  p50={record['p50_ms']:.2f} ms "
          f"p95={record['p95_ms']:.2f} ms "
          f"p99={record['p99_ms']:.2f} ms"
      )

  eligible = [x for x in results if "error" not in x]
  by_device: dict[int, list[float]] = {}
  for record in eligible:
    by_device.setdefault(record["physical_device"], []).append(record["p95_ms"])
  selected = min(
      by_device,
      key=lambda device: (statistics.fmean(by_device[device]), device),
  ) if by_device else None
  report = {
      "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "selected_device": selected,
      "precision_policy": {
          model: {
              "parameter_dtype": MODEL_PRECISION[model][0],
              "codec_compute_dtype": MODEL_PRECISION[model][1],
          }
          for model in models
      },
      "results": results,
  }
  target = output_dir / "cuda-matrix.json"
  temporary = target.with_suffix(".tmp")
  temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  temporary.replace(target)
  print(f"Saved {target}")
  return report


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--asset-root", required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--device", required=True, type=int)
  parser.add_argument("--warmup", type=int, default=50)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--parameter-dtype", default="float32", choices=("float32", "bfloat16"))
  parser.add_argument("--codec-compute-dtype", default="float32", choices=("float32", "bfloat16"))
  args = parser.parse_args()
  if not args.worker:
    parser.error("this module entry point is reserved for benchmark workers")
  result = _worker(
      args.asset_root, args.model, args.device, args.warmup, args.steps,
      args.parameter_dtype, args.codec_compute_dtype,
  )
  print(json.dumps(result))


if __name__ == "__main__":
  main()
