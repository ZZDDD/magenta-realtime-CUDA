# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""FastAPI coordinator and static web host."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
from pathlib import Path
import struct
import time
import uuid
import wave

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from magenta_rt.web.manager import WorkerManager, select_device
from magenta_rt.web.protocol import (
    BankCommand,
    COMMAND_ADAPTER,
    MidiCommand,
    ModelCommand,
    ParamsCommand,
    PromptsCommand,
    RecordingCommand,
    ResetCommand,
    TransportCommand,
)


FRAME_HEADER = struct.Struct("<4sIIIQ")
SAMPLE_RATE = 48000
FRAME_SAMPLES = 1920
CHANNELS = 2


class Runtime:
  def __init__(self, asset_root: str, model: str, device: int):
    self.asset_root = Path(asset_root).resolve()
    self.model = model
    self.device = device
    self.manager = WorkerManager(str(self.asset_root), device)
    self.websocket: WebSocket | None = None
    self.send_lock = asyncio.Lock()
    self.pump_task = None
    self.stage = "starting"
    self.error = None
    self.revision = 0
    self.frame_times = deque(maxlen=500)
    self.last_sequence = 0
    self.recording_id = None
    self.recording_pcm = bytearray()
    self.recording_started = None
    self.audio_prompts: dict[str, Path] = {}
    self.load_future: asyncio.Future | None = None

  async def send_json(self, payload: dict) -> None:
    if self.websocket is None:
      return
    async with self.send_lock:
      await self.websocket.send_json(payload)

  async def send_bytes(self, payload: bytes) -> None:
    if self.websocket is None:
      return
    async with self.send_lock:
      await self.websocket.send_bytes(payload)

  def status(self) -> dict:
    ordered = sorted(self.frame_times)
    p95 = ordered[int((len(ordered) - 1) * 0.95)] if ordered else None
    return {
        "model": self.model,
        "device": self.device,
        "stage": self.stage,
        "ready": self.stage == "ready",
        "error": self.error,
        "controller_connected": self.websocket is not None,
        "last_sequence": self.last_sequence,
        "frame_ms": self.frame_times[-1] if self.frame_times else None,
        "p95_ms": p95,
        "recording": self.recording_id is not None,
        "worker_id": self.manager.worker_id,
    }

  async def pump(self) -> None:
    while True:
      event = await asyncio.to_thread(self.manager.get_event, 0.5)
      if event is None:
        await asyncio.sleep(0)
        continue
      if event.get("worker_id") != self.manager.worker_id:
        continue
      kind = event.get("kind")
      if kind == "audio":
        self.last_sequence = int(event["sequence"])
        elapsed = float(event["elapsed_ms"])
        self.frame_times.append(elapsed)
        pcm = event["pcm"]
        if self.recording_id is not None:
          self.recording_pcm.extend(pcm)
        header = FRAME_HEADER.pack(
            b"MRT2",
            self.last_sequence,
            int(event["revision"]),
            FRAME_SAMPLES,
            int(event["generated_ns"]),
        )
        try:
          await self.send_bytes(header + pcm)
        except (RuntimeError, WebSocketDisconnect):
          self.websocket = None
      elif kind == "metrics":
        try:
          await self.send_json({"type": "metrics", **self.status()})
        except (RuntimeError, WebSocketDisconnect):
          self.websocket = None
      elif kind == "status":
        self.stage = event.get("stage", self.stage)
        self.error = None
        if "startup_seconds" in event:
          self.frame_times.clear()
          self.last_sequence = 0
        if self.stage == "ready" and self.load_future is not None:
          if not self.load_future.done():
            self.load_future.set_result(event)
        try:
          await self.send_json({"type": "status", **self.status(), **event})
        except (RuntimeError, WebSocketDisconnect):
          self.websocket = None
      elif kind == "error":
        self.stage = "error"
        self.error = event.get("message")
        if self.load_future is not None and not self.load_future.done():
          self.load_future.set_exception(RuntimeError(self.error))
        try:
          await self.send_json({"type": "error", "message": self.error})
        except (RuntimeError, WebSocketDisconnect):
          self.websocket = None

  async def switch_model(self, model: str) -> None:
    if model == self.model and self.stage == "ready":
      return
    self.stage = "switching"
    await self.send_json({"type": "status", **self.status()})
    previous = self.model
    try:
      self.model = model
      self.frame_times.clear()
      self.last_sequence = 0
      self.stage = "starting"
      self.load_future = asyncio.get_running_loop().create_future()
      worker_id = self.manager.reserve_worker_id()
      await asyncio.to_thread(self.manager.start, model, worker_id)
      await asyncio.wait_for(asyncio.shield(self.load_future), timeout=600)
    except Exception as exc:
      failure = f"Could not load {model}: {exc}"
      self.model = previous
      self.stage = "restoring"
      self.error = failure
      await self.send_json({"type": "error", "message": failure})
      self.load_future = asyncio.get_running_loop().create_future()
      worker_id = self.manager.reserve_worker_id()
      await asyncio.to_thread(self.manager.start, previous, worker_id)
      try:
        await asyncio.wait_for(asyncio.shield(self.load_future), timeout=600)
      except Exception as restore_exc:
        self.stage = "error"
        self.error = f"{failure}; restore failed: {restore_exc}"
        await self.send_json({"type": "error", "message": self.error})
    finally:
      self.load_future = None

  def finish_recording(self) -> str | None:
    if self.recording_id is None:
      return None
    recordings = self.asset_root / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)
    name = f"{self.recording_id}.wav"
    target = recordings / name
    with wave.open(str(target), "wb") as handle:
      handle.setnchannels(CHANNELS)
      handle.setsampwidth(2)
      handle.setframerate(SAMPLE_RATE)
      handle.writeframes(self.recording_pcm)
    self.recording_id = None
    self.recording_pcm.clear()
    self.recording_started = None
    return name


def create_app(asset_root: str, model: str, device: int) -> FastAPI:
  runtime = Runtime(asset_root, model, device)

  @asynccontextmanager
  async def lifespan(_app):
    runtime.manager.start(model)
    runtime.pump_task = asyncio.create_task(runtime.pump())
    try:
      yield
    finally:
      if runtime.pump_task:
        runtime.pump_task.cancel()
      runtime.manager.stop()

  app = FastAPI(title="Magenta RT Web", lifespan=lifespan)
  app.state.runtime = runtime

  @app.middleware("http")
  async def isolation_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cache-Control"] = "no-store"
    return response

  @app.get("/api/status")
  async def status():
    return runtime.status()

  @app.get("/api/config")
  async def config():
    return {
        "models": ["mrt2_base", "mrt2_small"],
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "frame_samples": FRAME_SAMPLES,
            "sample_format": "s16le",
            "header_bytes": FRAME_HEADER.size,
        },
        "limits": {"prompts": 6, "audio_upload_bytes": 50 * 1024 * 1024},
    }

  @app.get("/api/benchmarks")
  async def benchmarks():
    path = runtime.asset_root / "benchmarks" / "cuda-matrix.json"
    if not path.is_file():
      return {"selected_device": runtime.device, "results": []}
    return json.loads(path.read_text(encoding="utf-8"))

  @app.post("/api/prompts/audio")
  async def audio_prompt(file: UploadFile = File(...)):
    contents = await file.read(50 * 1024 * 1024 + 1)
    if len(contents) > 50 * 1024 * 1024:
      raise HTTPException(413, "audio prompt exceeds 50 MiB")
    suffix = Path(file.filename or "prompt.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}:
      raise HTTPException(415, "unsupported audio format")
    uploads = runtime.asset_root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(contents)
    prompt_id = path.stem
    runtime.audio_prompts[prompt_id] = path
    return {"id": prompt_id, "label": file.filename or "Audio prompt"}

  @app.get("/api/recordings/{recording_name}")
  async def recording(recording_name: str):
    if Path(recording_name).name != recording_name or not recording_name.endswith(".wav"):
      raise HTTPException(400, "invalid recording name")
    path = runtime.asset_root / "recordings" / recording_name
    if not path.is_file():
      raise HTTPException(404, "recording not found")
    return FileResponse(path, media_type="audio/wav", filename=recording_name)

  @app.websocket("/api/session")
  async def session(websocket: WebSocket):
    if runtime.websocket is not None:
      await websocket.close(code=4409, reason="another controller is connected")
      return
    await websocket.accept()
    runtime.websocket = websocket
    await runtime.send_json({"type": "status", **runtime.status()})
    try:
      while True:
        raw = await websocket.receive_json()
        try:
          command = COMMAND_ADAPTER.validate_python(raw)
        except ValidationError as exc:
          await runtime.send_json({
              "type": "error",
              "message": "invalid command",
              "details": exc.errors(include_url=False),
          })
          continue
        runtime.revision += 1
        revision = runtime.revision
        if isinstance(command, ModelCommand):
          await runtime.switch_model(command.model)
        elif isinstance(command, RecordingCommand):
          if command.action == "start":
            runtime.recording_id = uuid.uuid4().hex
            runtime.recording_pcm.clear()
            runtime.recording_started = time.time()
            await runtime.send_json({"type": "recording", "recording": True})
          else:
            name = runtime.finish_recording()
            await runtime.send_json({
                "type": "recording",
                "recording": False,
                "url": f"/api/recordings/{name}" if name else None,
            })
        elif isinstance(command, TransportCommand):
          runtime.manager.send({
              "type": "transport", "playing": command.playing,
              "revision": revision,
          })
        elif isinstance(command, ParamsCommand):
          runtime.manager.send({
              "type": "params",
              "params": command.model_dump(exclude={"type"}),
              "revision": revision,
          })
        elif isinstance(command, PromptsCommand):
          prompts = []
          invalid_source = None
          for item in command.prompts:
            value = item.model_dump()
            if item.source is not None:
              source = runtime.audio_prompts.get(item.source)
              if source is None:
                invalid_source = item.source
                break
              value["source"] = str(source)
            prompts.append(value)
          if invalid_source is not None:
            await runtime.send_json({
                "type": "error",
                "message": f"unknown audio prompt: {invalid_source}",
            })
            continue
          runtime.manager.send({
              "type": "prompts",
              "prompts": prompts,
              "revision": revision,
          })
        elif isinstance(command, MidiCommand):
          runtime.manager.send({
              "type": "midi", "note": command.note, "on": command.on,
              "revision": revision,
          })
        elif isinstance(command, ResetCommand):
          runtime.manager.send({"type": "reset", "revision": revision})
        elif isinstance(command, BankCommand):
          runtime.manager.send({
              "type": f"bank_{command.action}", "index": command.index,
              "revision": revision,
          })
    except WebSocketDisconnect:
      pass
    finally:
      if runtime.websocket is websocket:
        runtime.websocket = None
        try:
          runtime.manager.send({"type": "transport", "playing": False})
          runtime.manager.send({"type": "all_notes_off"})
        except RuntimeError:
          pass

  frontend = Path(__file__).resolve().parents[2] / "examples" / "web" / "ui" / "dist"
  if frontend.is_dir():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
  else:
    @app.get("/")
    async def missing_frontend():
      return JSONResponse({
          "service": "Magenta RT Web",
          "status": "frontend not built",
          "build": "npm --prefix examples ci && npm --prefix examples run build",
      })
  return app


def run(asset_root: str, model: str, device: str, host: str, port: int) -> None:
  import uvicorn

  selected = select_device(asset_root, device)
  uvicorn.run(
      create_app(asset_root=asset_root, model=model, device=selected),
      host=host,
      port=port,
      log_level="info",
  )
