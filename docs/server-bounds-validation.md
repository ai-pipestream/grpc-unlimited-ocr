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

## Actual file verification receipt

Rechecked on 2026-11-06 at 23:59 UTC on krick. The directory used by the
real-weight metadata probe is `/work/models/unlimited-ocr-2`. SHA256 checks
against every entry in models/MANIFEST succeeded for all 8 files, including
both safetensors shards. The recheck log is
`/tmp/uocr-bounds-model-hashes-recheck.log`. This verifies file contents,
not just sizes. A deployment mounting a different directory must repeat
this verification against that directory.

- Model: `baidu/Unlimited-OCR-2`
- Upstream revision: `5632329a32e103d5f1add270681e0f5c1233c6e4`
- Probe backend: `reference`
- Probe execution precision: `fp32`
- Probe device: `cpu (fp32 reference)`
- MANIFEST SHA256: `858dd87a290712f285a670190e160695e902e258e74bdedd2c6fceca90ed33e1`

Fixture identities for the prepared inference tests:

- `tests/goldens/quickstart-p2/page.png`: `639d6d2006313ac89d99e85f3f00038abec91e281f2975f28db28ffce97ed139`
- `tests/goldens/quickstart-p2/output_raw.txt`: `25f3819e16a327d3965758d97dbb9313c0532e380d4bda7a559ba6340526091e`
- `tests/goldens/quickstart-p2/tokens.txt`: `0bb7ecf74053c0f79c3431d767245976d2abd83b42e4b9429cc5c48a7aa78227`

The fixture comes from page 2 of the quickstart sample, rendered at 150 dpi.
The two-page transport test repeats that raster. The metadata probe does
not run inference on this fixture. No CUDA inference result is recorded.

`GetServiceInfoResponse.manifest_sha256` already describes the manifest
used at build time. CMake computes that value from the repository file.
The service does not verify mounted file hashes during startup; architecture
checks do not substitute for that verification. The successful file checks
above are separate evidence and must not be inferred from this RPC field.
No wire changes were needed for this receipt.
