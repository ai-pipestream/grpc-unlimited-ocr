# grpc-unlimited-ocr

A gRPC collector that runs the Unlimited-OCR 2 document model and streams
typed layout blocks and per-page `ai.pipestream.document.v1.Document`
fragments back to the caller. The model runs on a native engine in this
repo: weights load directly from the upstream safetensors, the forward
pass is C++. Implemented backends are the fp32 CPU reference and CUDA
(bf16 and int8). Intel SYCL/oneDNN, Apple Metal, Hailo offload and int4
remain planned, not selectable working backends. There is no inference runtime and no
Python at build or run time.

**Status:** the reference engine matches the upstream implementation token
for token, and the CUDA backend runs a page end to end (on an RTX 4080
SUPER: a historical single-page CLI measurement recorded 0.3 s of prefill
and about 285 decode tokens per second, not gRPC end-to-end throughput).
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

## Server resource limits

`GRPC_UOCR_MAX_PAGE_PIXELS` defaults to 40 million pixels and is checked
against the raster header before image allocation. Supplied page dimensions
must agree with that header. Compressed input caps still apply separately.

Each decode window must fit `GRPC_UOCR_MAX_STREAM_BUFFERED_BYTES`. If it
cannot, the RPC returns `RESOURCE_EXHAUSTED`; it does not exceed the cap or
wait for an incomplete window to free itself. Global raster admission is
also fail-fast when `GRPC_UOCR_MAX_BUFFERED_BYTES` is full. Callers can retry
with a smaller window, smaller rasters, or less concurrent work. Duplicate
page numbers are rejected even after their original window was consumed.

`GRPC_UOCR_MAX_OUTPUT_BUFFERED_BYTES` defaults to 16 MiB. It bounds queued
responses using protobuf's `SpaceUsedLong()` estimate plus object overhead;
a slow reader blocks the producer. A document slot is retained until the
writer finishes, so slow readers cannot accumulate queues beyond the
configured document concurrency. One additional response per active document
may be in the transport, and the model,
current page tokens, and optional folded Document have separate memory
costs. An individual response larger than the queue cap fails the RPC with
`RESOURCE_EXHAUSTED` when no transport cancellation is needed.

RPC admission is fail-fast before the initial read and worker creation.
The slot remains reserved through worker and watchdog shutdown. Clients
need not half-close after sending the declared number of pages; extra
messages after that boundary are not consumed.

`GRPC_UOCR_TRANSPORT_IDLE_MS` defaults to 30000. It limits blocked reads and
writes, not total request duration. Input-idle timing pauses during native
inference, including prefill; write-stall timing is independent. Internal
failure allows queued events 200 ms of stalled-operation grace before
interrupting a blocked transport operation with `TryCancel`. On the wire,
that operation returns `CANCELLED`, even if the handler has a more specific
failure status. Existing typed `PageFailure` events and partial results are
delivered when possible, but cannot be guaranteed to an unreadable client.
Cancellation releases blocked producers; native encode/prefill remains
non-preemptible, so compute leases can remain until that call returns.

The reference backend computes and reports `fp32`. With
`GRPC_UOCR_BACKEND=reference`, leave `GRPC_UOCR_WEIGHT_FORMAT` unset or use
`fp32`; other values are refused. CUDA supports `bf16` and `int8`.

The opt-in `server_model_stream_test` exercises real weights over gRPC with
one page, two pages, known fixture text and geometry, delayed reads, and
cancellation. It skips without
`UOCR_MODELS_DIR`; run it on an available device, not beside busy GPU jobs.
The scripted transport tests and queue tests do not prove model inference.
