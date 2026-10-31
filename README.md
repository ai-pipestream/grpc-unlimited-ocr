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
The gRPC server (`ai.pipestream.unlimitedocr.v1.UnlimitedOcrService`, port
50073) streams every decode step, the parsed blocks, per-page `Document`
fragments and the folded `Document`.
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

## The service

The contract is [`proto/ai/pipestream/unlimitedocr/v1/unlimited_ocr.proto`](proto/ai/pipestream/unlimitedocr/v1/unlimited_ocr.proto)
(buf-managed; `buf lint` and `buf format` run in CI). The client sends
`ParseOptions`, then one `PageImage` per page, then half-closes; the server
answers with `WindowStarted`, `Block`, `ContinuationLink`, `PageFailure`,
`PageTokens` (every decode step, always), `PageDocument`, the folded
`Document` and `ParseComplete`. `document.proto` and the OpenNLP protos it
imports are vendored byte-identical from gRParse; change them there.

```sh
docker run --rm --gpus all --read-only -p 50073:50073 \
  -v /path/to/unlimited-ocr-2:/models:ro grpc-unlimited-ocr:local
grpcurl -plaintext localhost:50073 ai.pipestream.unlimitedocr.v1.UnlimitedOcrService/GetServiceInfo
```

Configuration is `GRPC_UOCR_*` environment variables (see
[`src/service/config.h`](src/service/config.h)): `LISTEN_ADDRESS`
(`0.0.0.0:50073`), `BACKEND` (`cuda`, `reference`; `intel-gpu`, `metal` and
`cpu` once built), `DEVICE_INDEX`, `WEIGHT_FORMAT` (`bf16`, `int8`),
`MODELS_DIR` (`/models`), `PAGE_WINDOW` (10),
`MAX_TOKENS_PER_PAGE` (20000), `NGRAM_SIZE` (35), `NGRAM_WINDOW` (1024),
`NGRAM_WINDOW_SINGLE_PAGE` (128), `MAX_PAGES`, `MAX_PAGE_BYTES`,
`MAX_STREAM_BUFFERED_BYTES`, `MAX_BUFFERED_BYTES` and
`DOCUMENTS_IN_FLIGHT` (1). The server is built by default;
`-DUOCR_WITH_SERVER=OFF` builds only the engine (no gRPC).

## Licence

Apache-2.0. The model weights and the upstream code this engine ports are
MIT-licensed by Baidu; see [`NOTICE`](NOTICE). Upstream does not state the
provenance of its training data.
