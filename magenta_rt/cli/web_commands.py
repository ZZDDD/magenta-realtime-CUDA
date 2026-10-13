# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CLI commands for the CUDA-backed web application."""

import json

import click

from magenta_rt.cli import main


@main.group()
def web():
  """Run and qualify the browser-based MRT2 host."""


@web.command("download-assets")
@click.option("--asset-root", required=True, type=click.Path(path_type=str))
def download_assets_command(asset_root):
  """Download both CUDA checkpoints and shared resources at a pinned revision."""
  from pathlib import Path
  from huggingface_hub import snapshot_download
  from magenta_rt.web.assets import ASSETS, HF_REPOSITORY, HF_REVISION

  snapshot_download(
      repo_id=HF_REPOSITORY,
      revision=HF_REVISION,
      allow_patterns=list(ASSETS),
      local_dir=str(Path(asset_root).expanduser()),
  )
  click.echo("Download complete. Run 'mrt web verify-assets' to check integrity.")


@web.command()
@click.option("--asset-root", required=True, type=click.Path(path_type=str))
@click.option("--model", default="mrt2_base", type=click.Choice(["mrt2_base", "mrt2_small"]))
@click.option("--device", default="auto", help="Physical CUDA index or 'auto'.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def serve(asset_root, model, device, host, port):
  """Serve the web UI and one isolated CUDA model worker."""
  from magenta_rt.web.server import run

  run(asset_root=asset_root, model=model, device=device, host=host, port=port)


@web.command("benchmark")
@click.option("--asset-root", required=True, type=click.Path(path_type=str))
@click.option("--models", default="mrt2_base,mrt2_small", show_default=True)
@click.option("--devices", default="0,1", show_default=True)
@click.option("--warmup", default=50, type=int, show_default=True)
@click.option("--steps", default=1000, type=int, show_default=True)
def benchmark_command(asset_root, models, devices, warmup, steps):
  """Benchmark model/device pairs using production one-frame synchronization."""
  from magenta_rt.web.benchmark import run_matrix

  run_matrix(
      asset_root=asset_root,
      models=[x.strip() for x in models.split(",") if x.strip()],
      devices=[int(x.strip()) for x in devices.split(",") if x.strip()],
      warmup=warmup,
      steps=steps,
  )


@web.command("compare-precision")
@click.option("--asset-root", required=True, type=click.Path(path_type=str))
@click.option("--model", default="mrt2_base", type=click.Choice(["mrt2_base", "mrt2_small"]))
@click.option("--device", default=0, type=int, show_default=True)
@click.option("--frames", default=250, type=int, show_default=True)
def compare_precision_command(asset_root, model, device, frames):
  """Render FP32 and BF16 WAVs and compare objective signal statistics."""
  from magenta_rt.web.quality import compare_precision

  report = compare_precision(asset_root, model, device, frames)
  click.echo(json.dumps(report, indent=2))


@web.command("verify-assets")
@click.option("--asset-root", required=True, type=click.Path(path_type=str))
def verify_assets_command(asset_root):
  """Verify downloaded MRT2 resources against the published LFS manifest."""
  from magenta_rt.web.assets import verify_assets

  failures = verify_assets(asset_root, verbose=True)
  if failures:
    raise click.ClickException(f"{len(failures)} asset verification failure(s)")
