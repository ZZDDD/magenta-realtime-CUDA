# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Isolated, single-GPU JAX generation worker."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import time


def worker_main(command_queue, event_queue, asset_root: str,
                model_name: str, physical_device: int, worker_id: int,
                parameter_dtype: str, codec_compute_dtype: str) -> None:
  os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
  os.environ["MAGENTA_RT_HOME"] = asset_root
  os.environ["JAX_COMPILATION_CACHE_DIR"] = str(Path(asset_root) / "jax-cache")
  os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
  os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

  try:
    import jax
    import numpy as np

    from magenta_rt import MagentaRT2Jax, paths
    from magenta_rt.audio import Waveform
    from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

    if jax.default_backend() != "gpu":
      raise RuntimeError(
          "CUDA is unavailable. Install the CUDA JAX extra and check nvidia-smi."
      )
    paths.set_magenta_home(asset_root)
    cache_dir = Path(asset_root) / "jax-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
      jax.config.update("jax_compilation_cache_dir", str(cache_dir))
      jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
      jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    except AttributeError:
      pass

    def emit(event):
      event_queue.put({"worker_id": worker_id, **event})

    emit({"kind": "status", "stage": "loading", "model": model_name})
    started = time.perf_counter()
    model = MagentaRT2Jax(
        size=model_name,
        parameter_dtype=parameter_dtype,
        codec_compute_dtype=codec_compute_dtype,
    )
    emit({"kind": "status", "stage": "embedding", "model": model_name})

    embedding_cache = {}
    default_text = "disco funk"
    embedding_cache[default_text] = model.embed_style(default_text, use_mapper=True)
    prompt_embedding = embedding_cache[default_text]
    active_notes = np.zeros(128, dtype=np.bool_)
    onset_notes = np.zeros(128, dtype=np.bool_)
    params = {
        "temperature": 1.3,
        "top_k": 40,
        "cfg_musiccoca": 3.0,
        "cfg_notes": 1.0,
        "cfg_drums": 1.0,
        "drumless": False,
        "onset_mode": False,
        "unmask_width": 0,
    }
    state = model.initial_state()
    banks = [None, None, None]
    prepared = None
    dirty = True
    playing = False
    revision = 0
    sequence = 0
    stop = False
    next_deadline = time.monotonic()

    def rebuild():
      notes = np.full(128, -1, dtype=np.int32)
      active_indices = np.flatnonzero(active_notes)
      width = int(params["unmask_width"])
      if width >= 127:
        notes.fill(0)
      else:
        for note in active_indices:
          start = max(0, int(note) - width)
          end = min(128, int(note) + width + 1)
          notes[start:end] = 0
      if params["onset_mode"]:
        notes[active_indices] = np.where(onset_notes[active_indices], 2, 1)
      else:
        notes[active_indices] = 3
      conditioning = {
          MUSICCOCA.key: prompt_embedding,
          PIANOROLL_WITH_ONSETS.key: notes,
          DRUM_PIANOROLL.key: [0 if params["drumless"] else -1],
      }
      return model.prepare_conditioning(
          conditioning,
          cfg_scales={
              "musiccoca": params["cfg_musiccoca"],
              "notes": params["cfg_notes"],
              "drums": params["cfg_drums"],
          },
          temperature=params["temperature"],
          top_k=params["top_k"],
      )

    def set_prompts(items):
      nonlocal prompt_embedding
      active = [item for item in items if item["text"].strip() and item["weight"] > 0]
      if not active:
        prompt_embedding = np.zeros(768, dtype=np.float32)
        return
      embeddings = []
      weights = []
      for item in active:
        text = item["text"].strip()
        source = item.get("source") or text
        if source not in embedding_cache:
          if item.get("source"):
            waveform = Waveform.from_file(source)
            embedding_cache[source] = model.embed_style(waveform)
          else:
            embedding_cache[source] = model.embed_style(text, use_mapper=True)
        embeddings.append(embedding_cache[source])
        weights.append(item["weight"])
      normalized = np.asarray(weights, dtype=np.float32)
      normalized /= normalized.sum()
      prompt_embedding = np.sum(
          np.stack(embeddings) * normalized[:, None], axis=0
      ).astype(np.float32)

    def handle(command):
      nonlocal playing, state, dirty, revision, stop, next_deadline
      kind = command.get("type")
      revision = int(command.get("revision", revision))
      if kind == "stop":
        stop = True
      elif kind == "transport":
        playing = bool(command["playing"])
        if playing:
          next_deadline = time.monotonic()
      elif kind == "params":
        params.update(command["params"])
        dirty = True
      elif kind == "prompts":
        emit({"kind": "status", "stage": "prompt_embedding", "model": model_name})
        set_prompts(command["prompts"])
        dirty = True
        emit({"kind": "status", "stage": "ready", "model": model_name})
      elif kind == "midi":
        note = int(command["note"])
        if command["on"]:
          if not active_notes[note]:
            onset_notes[note] = True
          active_notes[note] = True
        else:
          active_notes[note] = False
          onset_notes[note] = False
        dirty = True
      elif kind == "all_notes_off":
        active_notes.fill(False)
        onset_notes.fill(False)
        dirty = True
      elif kind == "reset":
        state = model.initial_state()
      elif kind == "bank_save":
        banks[int(command["index"])] = jax.tree.map(
            lambda value: value.copy() if hasattr(value, "copy") else value,
            state,
        )
      elif kind == "bank_load" and banks[int(command["index"])] is not None:
        state = jax.tree.map(
            lambda value: value.copy() if hasattr(value, "copy") else value,
            banks[int(command["index"])],
        )

    emit({
        "kind": "status",
        "stage": "ready",
        "model": model_name,
        "device": physical_device,
        "startup_seconds": time.perf_counter() - started,
        "parameter_dtype": parameter_dtype,
        "codec_compute_dtype": codec_compute_dtype,
    })

    while not stop:
      if not playing:
        handle(command_queue.get())
        next_deadline = time.monotonic()
        continue
      try:
        while True:
          handle(command_queue.get_nowait())
      except queue.Empty:
        pass
      if stop or not playing:
        continue
      if dirty or prepared is None:
        prepared = rebuild()
        dirty = False
      tick = time.perf_counter_ns()
      pcm, state = model.stream_step(prepared, state)
      elapsed_ms = (time.perf_counter_ns() - tick) / 1_000_000
      sequence += 1
      emit({
          "kind": "audio",
          "sequence": sequence,
          "revision": revision,
          "generated_ns": time.monotonic_ns(),
          "elapsed_ms": elapsed_ms,
          "pcm": pcm.tobytes(),
      })
      onset_indices = np.flatnonzero(onset_notes)
      if len(onset_indices):
        onset_notes[onset_indices] = False
        dirty = True
      if sequence % 10 == 0:
        emit({
            "kind": "metrics",
            "sequence": sequence,
            "frame_ms": elapsed_ms,
        })
      next_deadline += 0.04
      delay = next_deadline - time.monotonic()
      while delay > 0 and playing and not stop:
        try:
          handle(command_queue.get(timeout=delay))
        except queue.Empty:
          break
        delay = next_deadline - time.monotonic()
      if delay < -0.2:
        next_deadline = time.monotonic()
  except BaseException as exc:  # Worker failures must reach the coordinator.
    event_queue.put({
        "worker_id": worker_id,
        "kind": "error",
        "message": f"{type(exc).__name__}: {exc}",
    })
    raise
