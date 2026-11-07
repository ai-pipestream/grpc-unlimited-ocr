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

The pinned gRPC 1.83.0 build completed successfully, including regenerated
protobuf sources, the server executable and every project test executable.
The project suite passed 27 tests, with 0 failures and 6 model-dependent
skips (33 registered). Both slow-reader regressions ran in this suite.
The build used GCC 15.3, Debug and UOCR_WERROR=ON without permissive flags.
Upstream gRPC emitted warnings, including a reflection serialization return
value warning; these were not first-party errors.
Logs: /tmp/uocr-bounds-pinned-build.log,
/tmp/uocr-pinned-project-build.log and /tmp/uocr-pinned-project-tests.log.
The default all target also builds unused gRPC variants, so the final build
named the server and all project test executables explicitly.

CUDA inference was not run: the RTX 4080 SUPER remained at 100% utilization
and about 9.6 GiB allocated, with other projects active. No jobs were stopped.
server_model_stream_test prepares one-page and two-page inference, delayed
reads, and cancellation. The reference run described below completed these
cases; no CUDA inference receipt is claimed.

The response limit uses protobuf SpaceUsedLong plus message overhead.
It does not measure total process memory: tensors, transport buffers,
current page traces, and the optional combined Document consume additional
memory. Cross-window KV carryover and gRParse wiring remain separate work.
Cancellation releases blocked output producers, but the native engine checks
the stop request at token callbacks. It does not interrupt an in-flight
vision encode or decoder prefill. The real-model cancellation test waits
for WindowStarted after prefill; it does not prove prefill cancellation
latency. Adding cancellation points inside native model execution remains
separate work.

The parent approved the full canonical schema after reviewing all additions.
The vendored document.proto now matches gRParse-typed-review/document.proto
byte for byte, including source attributes, sheet table declarations,
document metadata, cell formula/style fields, and PageQuality.text_layer_replaced.
Buf breaking with the previous copy, lint and format all pass. The generated
service rebuild and all 12 targeted service/resource tests pass with installed
gRPC 1.51.1. Logs: /tmp/uocr-canonical-service-build.log and
/tmp/uocr-canonical-tests.log.

## CI trigger

Commit 4a0a32d was docs-only with a CI skip marker. Feature pushes alone do
not trigger this repository's workflow, which listens for main pushes and
pull requests. Code commit 84b0ef5 has no skip marker and was pushed to
Forgejo first, then GitHub. Opening GitHub PR 10 triggered run 42316141927
for 84b0ef515085df49bcf89cdaf180f4b2665d248e. No manual dispatch was used.
Forgejo PR 4 includes the original server history and the fixes for main.

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

## Real reference inference

Code: 84b0ef515085df49bcf89cdaf180f4b2665d248e. On 2026-11-07 UTC the
real-model server transport test completed with exit 0 in about 10 minutes
25 seconds, including model loading. This is a functional receipt, not a
throughput benchmark: the host was running other projects.

Build: Release, GCC 15.3, UOCR_WERROR=ON, installed gRPC 1.51.1 through the
local adapter described above. Execution: reference backend, fp32, two
reference threads, nice 10, 1200-second process timeout. The mounted model
revision, hashes and fixture identities are those recorded above. Sampled
RSS reached about 19 GiB; this is a sample, not a measured peak or memory cap.

```sh
UOCR_MODELS_DIR=/work/models/unlimited-ocr-2 GRPC_UOCR_BACKEND=reference \
  timeout --kill-after=10s 1200s nice -n 10 stdbuf -oL \
  build-reference/server_model_stream_test
```

The one-page and two-page requests each returned nonempty token traces,
one page fragment per input, and successful ParseComplete over in-process
gRPC. Reads were delayed by 10 ms per event. A subsequent real-model request
was cancelled after WindowStarted; the client received CANCELLED and both
input-byte and document-slot leases returned to zero. The final log prints
`ok`. Log: /tmp/uocr-real-cpu-inference.log. This checks real inference and
transport completion; it does not compare generated text to the golden
tokens. Back-pressure with a fully stalled reader is covered separately by
the deterministic slow-reader regressions, not by the 10 ms delay alone.

`GetServiceInfoResponse.manifest_sha256` already describes the manifest
used at build time. CMake computes that value from the repository file.
The service does not verify mounted file hashes during startup; architecture
checks do not substitute for that verification. The successful file checks
above are separate evidence and must not be inferred from this RPC field.
No wire changes were needed for this receipt.

## Content and geometry qualification

The content check caught an error missed by transport-only checks: page 2's
final caption was an image_c block without coordinates. Actual before/after
model output is in docs/receipts/repeated-page-before.txt and
 docs/receipts/repeated-page-after.txt.

N-gram suppression crossed page boundaries and changed valid repeated
content. Commit ead223c adds the failing unit regression and text/geometry
checks. The fix confines suppression to the current page, retaining loop
protection within that page. Logs: /tmp/uocr-page-repeat-red.log and
/tmp/uocr-page-repeat-green.log.

Real reference/fp32 one-page and two-page RPC inference now passes all 4
text anchors per page, positive-area raw and normalized boxes inside
0..1000, expected vertical placement, completion, and cancellation after
prefill. The process exited 0. This qualifies one known raster and the
repeated-page case only, not other documents or hardware/precision modes.
Exact token equality is not required.

The canonical schema was regenerated with pinned gRPC 1.83, followed by a
server and project-test rebuild. Logs:
/tmp/uocr-final-canonical-regenerated-build.log and
/tmp/uocr-content-fixed-pinned-build.log. At this milestone, 28 tests passed,
0 failed, and 6 model-dependent tests skipped (34 registered). Corrupted
text/geometry rejection tests passed. Actual inference used the separate
optimized reference build with installed gRPC 1.51.1.
