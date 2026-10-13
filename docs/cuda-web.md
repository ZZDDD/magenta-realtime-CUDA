# CUDA runtime guide

Start with the [README quick start](../README.md#quick-start). This page covers the runtime behavior and diagnostics.

## Runtime and precision

The FastAPI server handles the UI, controller WebSocket, uploads, and recordings. A child process runs JAX with exactly one physical CUDA device visible. Switching models stops and joins the previous worker before starting the next; a failed switch attempts to restore the previous model.

| Model | Parameters | Parameter storage | Codec compute |
| --- | --- | --- | --- |
| `mrt2_small` | 230M | FP32 | FP32 |
| `mrt2_base` | 2.4B | BF16 | BF16 |

Both produce 48 kHz stereo audio in 1,920-sample (40 ms) frames. Base’s reduced precision is a CUDA host policy; use `mrt web compare-precision` to compare the generated outputs. The browser buffers one frame for Small and two for Base before playback. Generation time, browser buffering, and network time all contribute to latency.

## Asset management

`mrt web download-assets` downloads the files listed in `magenta_rt/web/assets.py` at Hugging Face revision `010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc`. `mrt web verify-assets` verifies every required file and writes `asset-manifest.json` after success. About 12 GB covers both checkpoints and the shared CUDA resources.

Always verify the download afterward. Asset directories also hold `jax-cache/`, `benchmarks/`, `uploads/`, and `recordings/`. Uploaded audio is retained on the server; remove it from `uploads/` when you no longer need it. Model context banks are temporary and live in worker memory.

## Launch options

Use `mrt web serve --help` for launch options, including the asset root, model, selected device, listen address, and port. The worker isolates the selected device before importing JAX. Model switching releases the previous worker before starting the next.

The service binds to localhost by default and has no built-in authentication. For remote browser access, use an SSH tunnel or an authenticated HTTPS proxy with WebSocket forwarding and cross-origin isolation headers.

## Checks and diagnostics

The server terminal shows worker errors and startup progress. `GET /api/status` reports the current model, device, stage, frame time, and error. `GET /api/config` describes the audio format and upload limits. The coordinator test suite runs without a GPU or model download:

```bash
python -m pytest
```

For latency qualification, benchmark the service’s actual precision path and use enough warmup steps to exclude first-use compilation. Reports are written to `benchmarks/cuda-matrix.json` beneath the asset root. Review P95 as well as mean frame time; a mean below 40 ms does not alone guarantee smooth playback.

## Offline generation

The extracted JAX CLI can also generate a short WAV without a browser:

```bash
MAGENTA_RT_HOME="$HOME/models/magenta-realtime-2" \
mrt jax generate \
  --model mrt2_small --prompt "jazz piano trio" --duration 4
```

The result is written under `outputs/` in the asset root. The offline command uses the original JAX generation path, which can have different precision and memory use from the web worker.
