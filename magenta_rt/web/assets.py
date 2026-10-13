# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Published asset manifest and integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HF_REPOSITORY = "google/magenta-realtime-2"
HF_REVISION = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
HF_MIRROR = "https://hf-mirror.com"

# Values come from the Hugging Face LFS metadata for HF_REVISION.
ASSETS: dict[str, tuple[int, str]] = {
    "checkpoints/mrt2_base.safetensors": (9836760520, "60f3e813d9da4a41a166c734a3074e6d54254c2fc14b0817bad6b8d25cddc044"),
    "checkpoints/mrt2_small.safetensors": (1128840272, "5dd1cbc7c606c512c21de0bcb04d4818bf0a3b873d7cbb9d1556d67d3b034de3"),
    "resources/musiccoca/audio_preprocessor.tflite": (8729640, "656ca4c358451c2b85932e66efcfd2ba62492f4435953d775bfc1d3c08329a30"),
    "resources/musiccoca/mapper.tflite": (86166664, "2f9743cc8f121a588b69c7f4d79a2a4111ce81864cbde8830054cd5e97f3d717"),
    "resources/musiccoca/music_encoder.tflite": (370935584, "0d4501af799834e383d904c34ee826c61eb53c69682bf15a981a46c1bb32793a"),
    "resources/musiccoca/pretrained_vector_quantizer.tflite": (72422108, "7a8a19e2119ad405818eae84a331a970f1a582b3389d4bfd27814f75b455a444"),
    "resources/musiccoca/spm.model": (517448, "ff325a99b61ba5726cf6437cde6eefbb633dbaa363a684f7a97ed99b55202cca"),
    "resources/musiccoca/text_encoder.tflite": (418674324, "e1222e3418cbe8cc2623939571bae8e9ab6f0d511404b0d83da69f4e6e11b272"),
    "resources/spectrostream/decoder.safetensors": (209853216, "0ac6f100a24945fb434783fde6acd7902ceaa8bca492ca317edfc75dd51c42dd"),
    "resources/spectrostream/encoder.safetensors": (37013392, "f20c197ddcbb9cd43e1a97f9bee0d07d211f79966cd3862d9830a32885090f72"),
    "resources/spectrostream/quantizer.safetensors": (67108984, "0ba89dcb85344bb14f4f34b8f597c0d6adaa560002d4fd88879a2944c98a20f0"),
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def verify_assets(asset_root: str | Path, verbose: bool = False) -> list[str]:
  """Return descriptions of missing, truncated, or corrupt required assets."""
  root = Path(asset_root)
  failures = []
  for relative, (expected_size, expected_hash) in ASSETS.items():
    path = root / relative
    if not path.is_file():
      failures.append(f"missing: {relative}")
      continue
    actual_size = path.stat().st_size
    if actual_size != expected_size:
      failures.append(
          f"size mismatch: {relative} ({actual_size} != {expected_size})"
      )
      continue
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
      failures.append(f"sha256 mismatch: {relative}")
      continue
    if verbose:
      print(f"ok  {relative}")

  manifest = {
      "repository": HF_REPOSITORY,
      "revision": HF_REVISION,
      "endpoint": HF_MIRROR,
      "assets": {
          key: {"size": size, "sha256": sha}
          for key, (size, sha) in ASSETS.items()
      },
  }
  if not failures:
    root.mkdir(parents=True, exist_ok=True)
    (root / "asset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
  elif verbose:
    for failure in failures:
      print(f"FAIL {failure}")
  return failures
