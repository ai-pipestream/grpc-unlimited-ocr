# Server validation, 2026-11-06

Base: GitHub main `809e1b1903b2f0191882aee8b0554cbbe04bc0fc`.
Verified on Forgejo `recovery/github-server-809e1b1` before feature pushes.

## Reproduction

Commit `0fada71` contains the initial regression tests. All 6 failed before
implementation changes: incorrect reference precision, forgotten consumed
page numbers, global admission waiting for cancellation, stream admission
exceeding the configured limit, raster pixel checks, and output buffering.
The log is `/tmp/uocr-bounds-red-tests.log`.

A further multi-client test failed before changing document-slot lifetime:
a finished decoder released the slot while responses waited for a reader.
That log is `/tmp/uocr-bounds-slots-red.log`.

## Local verification

GCC 15.3, Debug, UOCR_WERROR=ON, installed gRPC 1.51.1 and system protoc.
A temporary FetchContent adapter in `/tmp/uocr-bounds-system-grpc` selects
installed libraries for this local build. The repository continues to use
gRPC 1.83.0. No permissive compiler flag was used.

The CPU suite covers admission, duplicate tracking, raster pixels, response
limits, retry after exhaustion, slow-reader back-pressure, cancellation,
and slot lifetime through response delivery. Model-dependent tests are
reported separately as skipped, never counted as inference proof.
Final CPU result: 27 passed, 0 failed, 6 skipped, 33 registered tests.

All 8 local model files match models/MANIFEST SHA256 values. Loading these
real weights on the reference backend and calling GetServiceInfo over an
in-process gRPC server reports reference, fp32, and cpu (fp32 reference).
This checks startup and metadata, not OCR inference:

```sh
UOCR_MODELS_DIR=/work/models/unlimited-ocr-2 GRPC_UOCR_BACKEND=reference \
  timeout 180s build-system/server_model_stream_test --info-only
```

Logs on krick:

- /tmp/uocr-bounds-final-build.log
- /tmp/uocr-bounds-final-tests.log
- /tmp/uocr-bounds-real-info.log
- /tmp/uocr-bounds-model-hashes.log
- /tmp/uocr-bounds-buf-lint.log

## Remaining evidence

The fresh gRPC 1.83.0 build has a separate log at
/tmp/uocr-bounds-pinned-build.log. A successful exit and subsequent tests
are required before calling that build verified.

CUDA inference was not run: the RTX 4080 SUPER remained at 100% utilization
and about 9.6 GiB allocated, with other projects active. No jobs were stopped.
server_model_stream_test prepares one-page and two-page inference, delayed
reads, and cancellation for an available device. These cases require a
CUDA build or a scheduled reference inference run.

The response limit uses protobuf SpaceUsedLong plus message overhead.
It does not measure total process memory: tensors, transport buffers,
current page traces, and the optional combined Document consume additional
memory. Cross-window KV carryover and gRParse wiring remain separate work.

The requested shared-schema copy was not applied because the source also
adds source attributes, sheet table declarations, document metadata, and
cell formula/style fields beyond PageQuality.text_layer_replaced.
