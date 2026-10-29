# grpc-unlimited-ocr

A gRPC collector that runs the Unlimited-OCR 2 document model and streams
typed layout blocks and per-page `ai.pipestream.document.v1.Document`
fragments back to the caller. The model runs on a native engine in this
repo: weights load directly from the upstream safetensors, the forward
pass is C++, and each hardware target is one backend written at that
target's lowest level (CUDA, Intel SYCL/oneDNN, Apple Metal, and a Hailo
offload for the vision encoder). There is no inference runtime and no
Python at build or run time.

**Status:** the reference engine matches the upstream implementation token
for token, and the CUDA backend runs a page end to end (on an RTX 4080
SUPER: 0.3 s of prefill per page, about 285 tokens per second end to end).
The gRPC server is not written yet.
The plan, wire contract and phases are in [`docs/design.md`](docs/design.md).

## Build and test

Requires CMake 3.25 or newer and a C++23 compiler (GCC 15).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests that need the model weights skip (exit 77) unless
`UOCR_MODELS_DIR` points at a directory holding the files listed in
[`models/MANIFEST`](models/MANIFEST):

```sh
UOCR_MODELS_DIR=/path/to/unlimited-ocr-2 ctest --test-dir build --output-on-failure
```

A green run proves nothing about a test that skipped; read the skip count.

With the CUDA backend (`-DUOCR_WITH_CUDA=ON`; with a CUDA 12 toolkit add
`-DCMAKE_CUDA_HOST_COMPILER=g++-13`), or in the image:

```sh
docker build -t grpc-unlimited-ocr:local .
docker run --rm --gpus all -v /path/to/unlimited-ocr-2:/models:ro -v "$PWD/pages:/pages:ro" \
  grpc-unlimited-ocr:local /pages/page1.png
```

## Licence

Apache-2.0. The model weights and the upstream code this engine ports are
MIT-licensed by Baidu; see [`NOTICE`](NOTICE). Upstream does not state the
provenance of its training data.
