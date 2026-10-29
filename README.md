# grpc-unlimited-ocr

A gRPC collector that runs the Unlimited-OCR 2 document model and streams
typed layout blocks and per-page `ai.pipestream.document.v1.Document`
fragments back to the caller. The model runs on a native engine in this
repo: weights load directly from the upstream safetensors, the forward
pass is C++, and each hardware target is one backend written at that
target's lowest level (CUDA, Intel SYCL/oneDNN, Apple Metal, and a Hailo
offload for the vision encoder). There is no inference runtime and no
Python at build or run time.

**Status:** phase 0, the reference engine. Nothing serves requests yet.
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

## Licence

Apache-2.0. The model weights and the upstream code this engine ports are
MIT-licensed by Baidu; see [`NOTICE`](NOTICE). Upstream does not state the
provenance of its training data.
