# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Lifecycle manager for the isolated JAX subprocess."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import queue

from magenta_rt.web.worker import worker_main


MODEL_PRECISION = {
    "mrt2_base": ("bfloat16", "bfloat16"),
    "mrt2_small": ("float32", "float32"),
}


class WorkerManager:
  def __init__(self, asset_root: str, device: int):
    self.asset_root = str(Path(asset_root).resolve())
    self.device = device
    self.context = multiprocessing.get_context("spawn")
    self.command_queue = None
    self.event_queue = None
    self.process = None
    self.model = None
    self.worker_id = 0
    self._exit_reported = False

  def reserve_worker_id(self) -> int:
    self.worker_id += 1
    return self.worker_id

  def start(self, model: str, worker_id: int | None = None) -> None:
    if worker_id is None:
      worker_id = self.reserve_worker_id()
    elif worker_id != self.worker_id:
      raise ValueError("reserved worker ID is no longer current")
    self.stop()
    self._exit_reported = False
    self.command_queue = self.context.Queue(maxsize=64)
    self.event_queue = self.context.Queue(maxsize=16)
    parameter_dtype, codec_compute_dtype = MODEL_PRECISION[model]
    self.process = self.context.Process(
        target=worker_main,
        args=(
            self.command_queue,
            self.event_queue,
            self.asset_root,
            model,
            self.device,
            worker_id,
            parameter_dtype,
            codec_compute_dtype,
        ),
        daemon=True,
        name=f"mrt-{model}-gpu{self.device}",
    )
    self.model = model
    self.process.start()

  def stop(self) -> None:
    if self.process is None:
      return
    if self.process.is_alive() and self.command_queue is not None:
      try:
        self.command_queue.put({"type": "stop"}, timeout=1)
      except Exception:
        pass
      self.process.join(timeout=10)
    if self.process.is_alive():
      self.process.terminate()
      self.process.join(timeout=5)
    self.process.close()
    if self.command_queue is not None:
      self.command_queue.close()
    if self.event_queue is not None:
      self.event_queue.close()
    self.process = None
    self.command_queue = None
    self.event_queue = None

  def send(self, command: dict) -> None:
    if self.command_queue is None:
      raise RuntimeError("model worker is not running")
    self.command_queue.put(command, timeout=2)

  def get_event(self, timeout: float = 1.0):
    if self.event_queue is None:
      return None
    try:
      return self.event_queue.get(timeout=timeout)
    except queue.Empty:
      if (
          self.process is not None
          and not self.process.is_alive()
          and not self._exit_reported
      ):
        self._exit_reported = True
        return {
            "worker_id": self.worker_id,
            "kind": "error",
            "message": f"model worker exited with code {self.process.exitcode}",
        }
      return None
    except (OSError, ValueError):
      # A blocking queue read may be interrupted by an intentional switch.
      return None


def select_device(asset_root: str, requested: str) -> int:
  if requested != "auto":
    return int(requested)
  report = Path(asset_root) / "benchmarks" / "cuda-matrix.json"
  if report.is_file():
    selected = json.loads(report.read_text(encoding="utf-8")).get("selected_device")
    if selected is not None:
      return int(selected)
  return 0
