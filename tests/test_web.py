# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Portable tests for the CUDA web coordinator."""

import asyncio
import struct
import wave

import numpy as np
import pytest
from pydantic import ValidationError

from magenta_rt import paths
from magenta_rt.web.manager import MODEL_PRECISION, select_device
from magenta_rt.web.protocol import COMMAND_ADAPTER, PromptsCommand
from magenta_rt.web.quality import _signal_stats
from magenta_rt.web.server import FRAME_HEADER, Runtime


def test_protocol_accepts_audio_source_and_rejects_extra_fields():
  command = COMMAND_ADAPTER.validate_python({
      "type": "prompts",
      "prompts": [{"text": "clip.wav", "weight": 0.5, "source": "abc123"}],
  })
  assert isinstance(command, PromptsCommand)
  assert command.prompts[0].source == "abc123"

  with pytest.raises(ValidationError):
    COMMAND_ADAPTER.validate_python({"type": "reset", "unexpected": True})


def test_binary_audio_header_layout():
  encoded = FRAME_HEADER.pack(b"MRT2", 7, 11, 1920, 123456789)
  assert len(encoded) == 24
  assert struct.unpack("<4sIIIQ", encoded) == (
      b"MRT2", 7, 11, 1920, 123456789
  )


def test_signal_stats_rejects_clipping_and_nonfinite_values():
  samples = np.zeros((4096, 2), dtype=np.int16)
  samples[:, 0] = 1000
  stats = _signal_stats(samples)
  assert stats["finite"]
  assert stats["clipped_fraction"] == 0
  assert 0 < stats["peak"] < 1


def test_exact_asset_root_and_precision_policy(tmp_path):
  previous = paths.magenta_home()
  try:
    paths.set_magenta_home(tmp_path)
    assert paths.checkpoints_dir() == tmp_path / "checkpoints"
  finally:
    paths.set_magenta_home(previous)

  assert MODEL_PRECISION["mrt2_base"] == ("bfloat16", "bfloat16")
  assert MODEL_PRECISION["mrt2_small"] == ("float32", "float32")


def test_select_device_uses_benchmark_report(tmp_path):
  benchmark_dir = tmp_path / "benchmarks"
  benchmark_dir.mkdir()
  (benchmark_dir / "cuda-matrix.json").write_text(
      '{"selected_device": 1}', encoding="utf-8"
  )
  assert select_device(str(tmp_path), "auto") == 1
  assert select_device(str(tmp_path), "0") == 0


def test_recording_writes_stereo_48khz_wav(tmp_path):
  runtime = Runtime(str(tmp_path), "mrt2_small", 0)
  runtime.recording_id = "test-recording"
  runtime.recording_pcm.extend(b"\x00\x00\x01\x00" * 32)
  assert runtime.finish_recording() == "test-recording.wav"

  target = tmp_path / "recordings" / "test-recording.wav"
  with wave.open(str(target), "rb") as handle:
    assert handle.getnchannels() == 2
    assert handle.getsampwidth() == 2
    assert handle.getframerate() == 48000
    assert handle.getnframes() == 32


class _ReadyManager:
  def __init__(self, runtime, fail_model=None):
    self.runtime = runtime
    self.fail_model = fail_model
    self.worker_id = 0
    self.started = []

  def reserve_worker_id(self):
    self.worker_id += 1
    return self.worker_id

  def start(self, model, worker_id=None):
    assert worker_id == self.worker_id
    self.started.append(model)
    future = self.runtime.load_future
    loop = future.get_loop()
    if model == self.fail_model:
      loop.call_soon_threadsafe(
          future.set_exception, RuntimeError("checkpoint rejected")
      )
    else:
      loop.call_soon_threadsafe(future.set_result, {"stage": "ready"})


def test_model_switch_waits_for_ready_and_rolls_back(tmp_path):
  async def run_test():
    runtime = Runtime(str(tmp_path), "mrt2_base", 1)
    manager = _ReadyManager(runtime, fail_model="mrt2_small")
    runtime.manager = manager
    await runtime.switch_model("mrt2_small")
    assert manager.started == ["mrt2_small", "mrt2_base"]
    assert runtime.model == "mrt2_base"

  asyncio.run(run_test())
