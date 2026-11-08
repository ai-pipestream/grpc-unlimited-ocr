# CUDA qualification, 2026-11-08

The RTX 4080 SUPER native CUDA backend passed real model and TCP service
tests. The tested ownership-fix source is
`bbf14dc9335f5bcc6938df10175b815100951293`, following partial-KV cleanup
`ceb5617` and per-prefill graph invalidation `665275d`. Parent's pure-virtual
reset interface was subsequently imported as `96a9ad0`; its incremental
CUDA decoder and generator recovery tests also passed.

## Device, runtime and inputs

- krick: RTX 4080 SUPER 16 GiB, driver 615.71.09, CUDA 12.4, Release,
  architecture 89, GCC 15 C++ and GCC 13 nvcc host compiler.
- Native project forward pass and CUDA kernels; cuBLAS for dense GEMM.
  No replacement inference runtime and no CPU fallback for a CUDA request.
- Model revision `5632329a32e103d5f1add270681e0f5c1233c6e4`, original
  `/work/models/unlimited-ocr-2` files. All eight manifest hashes passed.
- stb `003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5`; JSON
  `55f93686c01528224f448c19128836e7df245f72` (v3.12.0); gRPC
  `c876f4da50f7da2f331888b88b2a7243514139fe` (v1.83.0).

An earlier local stb override was wrong. It was replaced with the declared
pin before the retained full-model/RPC qualification. The Nano pins were
checked separately. Tiny numerical checks reject size mismatch, NaN and
infinity; intentionally masked attention negative infinity is handled
explicitly.

## Results

The post-fix full golden passed BF16 and INT8: exact expected text, structure
and EOS; six boxes with maximum coordinate differences of respectively zero
and one on the 0..1000 coordinate scale. Generation took 0.835519 and
0.641181 seconds in those single runs; these are not a paired benchmark.

On the same source, each format passed three actual TCP cancellation and
recovery cycles using the common `--cancel-only` harness. Recovery assertions
cover content, geometry, tokens, fragments, trailer, status and admission
release. The earlier full battery of six sustained requests plus three
cancellation/recovery cycles passed both formats before the reset changes;
those earlier results are retained under their earlier scope.

The explicit FP32 reference backend also completed six sustained TCP requests
and three cancellation/recovery cycles. Precision and background load differ
from the CUDA runs, so no clean paired speedup is claimed from those timings.

Fault tests passed five Context initialization acquisition boundaries and
both Graph handoff boundaries, with exact successful release counts, inactive
capture, idempotent reset and same-decoder recapture/replay matching reference
logits. Partial KV allocation failures passed three cycles per precision.
Compute Sanitizer reported zero errors, including stream-ordered race checks.

## Orin limitation

Nano1 is an 8 GiB Orin ARM64 host running CUDA 13.2 / driver 595.78. Three
native finite-checked CUDA kernel/model-component tests passed. Full-model
inference does **not** fit the tested 4400 MiB container profile alongside
the existing TEI service. After weights load, live CUDA pool allocations are
4,248,917,248 bytes, leaving insufficient workspace. A conservative 1 GiB
headroom check now exits with an error instead of entering the earlier OOM
path. It is an admission guard, not a tracked allocation cap or peak bound.

No full-model Orin RPC qualification or working-Jetson claim is made. The
container budget was not raised and TEI was not stopped. Nonzero explicit
device-memory caps are rejected for CUDA rather than silently ignored.

## Receipts

`/work/main/grpc-services/hardware-receipts/cuda-20261108/unlimited/` contains
raw logs, model hashes, build configuration, a source archive and binary
manifest. Frozen `bbf14dc` binaries are in `/tmp/uocr-qualified-bbf14dc/`;
the manifest was captured after tests and before rebuilding for `96a9ad0`.
It does not relabel old binaries as the new source.

The raw Nsight CUPTI trace is retained, but its matching importer failed on
event ordering. It is not an analyzed profiler result. Existing native CUDA
Docker paths carry these source changes; no new full container build or
publication is claimed by these host tests.
