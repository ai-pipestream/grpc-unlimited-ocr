# Native Intel B70 qualification

2026-11-08: the native model, single-page golden repeats and strict one/two-page
TCP RPC matrix pass on B70. No merge or deployment was performed.

Final private test source: `a2b9798e22e8eb367f30f0e11b0193c86f27e9a4`.
Native runtime source is identical to `a048ced90ffc5fdc7157546756a55250bf6b4690`;
the final launcher verifies that source diff is empty and the worktree clean.
Private binary SHA256:
`83be98338333a58b4106aa4294bd68d028319f46048e70388177293d5d4757ec`.
All eight original model files were reverified before the matrix; their
hashes and the golden/common/private binary hashes are in
[`receipts/intel-final-private-rpc-b70.txt`](receipts/intel-final-private-rpc-b70.txt).

The final matrix completed with exit 0 in 439.98 seconds: eight complete
RPCs (12 pages), plus three requests cancelled after real GPU prefill.
Every completed page passed its matching oracle's text, labels, structure
and geometry gate. The complete token IDs, raw output, serialized blocks
and validated document payloads repeated identically for each page count.
All input/document leases returned; cancellation and admission release took
110.611–140.827 ms. Anonymous RSS was 261,718,016 bytes at two-page warmup
and 268,468,224 bytes at the end, within the unchanged warmup +128 MiB bound.
Process peak RSS was 6,944,284 KiB, including mapped model files. The native
allocation cap remained 12 GiB. One-page RPCs took 26.53–27.31 seconds;
two-page RPCs took 77.11–79.56 seconds on the shared host.

At native source `a048ced90ffc5fdc7157546756a55250bf6b4690`, the full golden
passed three repetitions with identical features, all 171 token IDs, text and
six boxes. Every box matched the upstream single-page golden coordinates.
Vision feature error was 0.00002694 against the unchanged 0.058 gate. Native
retained allocation was 6,829,784,464 bytes and peak was 7,079,610,769 bytes,
under the 12 GiB cap. The final repetition took 26.412 seconds, including
22.569 seconds of decode (7.577 tokens/s). These are shared-host measurements,
not controlled throughput benchmarks. The receipt records source and binary
SHA256, all eight model checks, GPU submissions, timings and process memory.

The common TCP gate then passed six requests alternating one/two pages and
three in-flight cancellation/admission-release/recovery cycles: 12 completed
pages, plus three cancelled requests. The private gate passed its single-page
golden, then rejected the two-page title structure against that single-page
expectation. The RED receipt is preserved. A subsequent two-page reference
decoder run, using the committed upstream fp32 projector features, reproduced
the LEVEL1/LEVEL2 titles and printed text/boxes from Intel. The private gate
now selects that separate two-page oracle; the single-page oracle and both
geometry thresholds are unchanged. The final hardware rerun passes.

Independent review found that the original RPC harness counted document
fragments without checking their payloads. `model_document_checks.h` now
checks unique token/fragment page identities, complete text, pixel geometry,
body and caption references, and a single complete document. Validated typed
documents enter the repeat snapshot. Eleven host-only mutations pass,
including missing/empty/duplicate/swapped payloads, invalid references, text,
geometry and NaN. The strengthened private hardware gate passes; the host
assertion receipt is `receipts/intel-final-host-assertions.txt`.

At test source `5b52adda4e760c03e7bd374d41025811d9c71ee8`, B70 also passed
injected widening-buffer failure and matmul scratch failure after the
widening kernel was queued. Both preserved the original allocation exception,
restored the live-byte baseline, retained unchanged bf16 weights and recovered
identical projection output on the same context. This is focused fault
coverage, not a full-model out-of-memory claim. Native model source remains
unchanged from a048ced.

The results below belong to the earlier model checkpoint. Independent review
then found partial cache growth could retain allocations after failure, and
LayerNorm lost variance on large-offset inputs. The current correctness batch
adds the parent generator exception guard, an idempotent native cache reset,
and shifted, centered LayerNorm in a source compiled with precise FP.
Native failure/recovery and offset-row gates now pass on B70: injected partial
growth, a real 64 MiB allocation cap, and scratch failure after growth all
recover exact full logits and tokens on the same model. Retained bytes return
to baseline; repeated reset is harmless.
The first centered-kernel retry failed at width 768 under default FP; setting
precise FP for the entire library then failed the existing position-table
gate. Both RED logs are retained. Precise FP is now restricted to LayerNorm;
no oracle tolerance changed. Native/server builds and 23 CPU-only tests pass.

Pinned vision diagnostics found projector relative error 0.0536434 at
`7f2bd2f`, increasing to 0.0630154 at `43f7e8a`, above the unchanged 0.058
gate. That gate existed in the standalone vision test but was not exercised
with weights in the first checkpoint run; it was added to the full golden
test in `6ffeca7`. Keeping projection inputs in fp32 reduces SAM error to
0.00001784 and projector error to 0.00002694. Matrices stay bf16 in device
storage; a temporary GPU widening supplies each vision fp32 GEMM. The decoder
continues to use bf16 projection inputs. Tiny vision error is now 0.000000373.
Full golden repeats and the strict multi-page RPC matrix pass.

`intel_vision_diagnostic` records SAM/projector arrays and finite comparisons;
it is a diagnostic executable, not a replacement for the qualification gates.
One candidate run used a stale vision object after timestamp-preserving sync;
its receipt is explicitly rejected. A forced rebuild and symbol check verified
the fp32 projection call before the accepted diagnostic run. Subsequent source
syncs use content checks and do not preserve timestamps on changed files.

`UOCR_WITH_INTEL=ON` provides `intel-gpu` through Engine and uocr-run.
The selector accepts Intel Level Zero GPUs. CPU fallback is absent.
Weights stay in upstream safetensors; resident matrices use bf16 and accumulation
uses fp32. Vision GEMMs preserve fp32 inputs; decoder projections round inputs
to bf16. oneDNN is used exclusively for matmul. SYCL implements SAM/CLIP,
projector, decoder attention, RoPE, norms, convolution patch extraction,
position interpolation, entmax routing, and MoE selection and arithmetic.
Prefill expert selection also runs on the GPU.

Image preprocessing, prompt layout, token decoding and logit policy run on
the host. Intel model code calls no reference forward operation. The
existing typed protobuf stream remains unchanged.

`GRPC_UOCR_DEVICE_MEMORY_CAP` limits native USM and explicit matmul scratch.
Zero chooses 80% of device capacity. Tests select 12 GiB. Driver caches and
host memory are separate. Prefill cannot be interrupted; decode cancellation
and admission recovery pass the common real TCP test.

## Historical checkpoint results (7f2bd2f)

Arc Pro B70, oneAPI 2025.3.3, oneDNN 3.9, krick-1:

- All 8 model SHA256 checks pass.
- Numerical mutation tests reject NaN, infinities and length mismatches.
  Masks accept matching negative-infinity sentinels only.
- Hardware primitive tests pass, including bf16 widths 768/896/1280/2304/6848.
  Tiny tiled vision error: 0.00259. Tiny decoder: prefill and 44 steps pass
  at tolerance 0.02. Repeated GEMMs are exact.
- Real golden: 3 complete repetitions pass, with exact token IDs, text and
  box coordinates across requests. All match upstream fp32 text and labels;
  6 boxes differ from fp32 by at most 1/1000. Vision features repeat exactly.
- Retained native allocations: 6,829,784,464 bytes throughout. Peak:
  7,083,542,929 bytes. Actual GPU submissions are recorded for vision,
  decoder prefill and generation.

The earlier repeatability failure remains in `intel-native-golden-red.txt`.
The subsequent 3-repeat test enables oneDNN deterministic matmul and passes. The earlier run also used stale stb 2c980bb; the successful
rerun uses the required 003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5 revision.
JSON v3.12.0 and gRPC c876f4da50f7da2f331888b88b2a7243514139fe match the
repository pins. Model files passed SHA verification on the target host.

The complete retry took 3m35.6s under host load above 100 on 32 logical CPUs.
These timings do not establish controlled throughput. Earlier historical
runs recorded 7.67/7.73 decode tokens/s. Current measurements are listed above;
no controlled hardware speedup comparison is claimed.

The common RPC test includes d978a13a4dea8dd23ee8a7d5c4efb99323f1ed11:
6 repeated requests and 3 cancellation/recovery cycles through loopback TCP.
`intel_native_rpc_test` adds full golden content and geometry checks for
every page, exact token/text/block repeat identity and process memory checks.
Neither RPC test had run at the historical checkpoint. Current results are
listed above; the isolated native/server builds now pass.

The common harness also accepts `--cancel-only` from 48abee7. This has not
been used to skip the initial Intel request gate. The private golden checker
passes deliberate extra-text, changed-label, displaced-box and missing-block
mutations on the host. That checks assertions, not GPU inference.

## Commands

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-intel -G Ninja -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DUOCR_WITH_INTEL=ON
nice -n10 cmake --build build-intel -j1
export UOCR_INTEL_TEST_LOCK=/path/to/shared/model-test.lock
scripts/qualify-intel.sh build-intel tiny
scripts/qualify-intel.sh build-intel golden /path/to/unlimited-ocr-2
scripts/qualify-intel.sh build-intel rpc /path/to/unlimited-ocr-2
```

A scheduling grant is required before a GPU command. The script uses a
nonblocking lock, hash checks and finite timeouts. A skipped test fails the
script. Full-model modes require the `intel_memory_check` target, built when
Level Zero Sysman headers and loader are available. It refuses missing or
ambiguous telemetry and requires 12 GiB free. RPC mode runs the common
request/recovery gate followed by the private golden and memory gate, with
one model instance at a time and a 30-minute limit per executable.
`-DUOCR_WITH_SERVER=OFF` omits gRPC for standalone model testing.
The service accepts `GRPC_UOCR_BACKEND=intel-gpu` with bf16 precision and
`GRPC_UOCR_MODELS_DIR`. The runner accepts `--backend intel-gpu --weights bf16`.

## Scope and remaining limits

These receipts qualify the isolated Intel branch and the known page in
one/two-page requests. They do not qualify the parent integration build,
a container deployment, a broad document corpus, or a long-document soak.
The two-page decoder oracle reuses committed upstream fp32 vision features;
it is not a second full CPU vision run. Native inference always uses the real
Intel vision encoder. The cap covers native USM and explicit GEMM scratch,
not driver caches; anonymous host memory is checked separately. Cancellation
is tested after prefill, not during vision/prefill. No full-model OOM recovery
or controlled throughput/speedup claim is made. Earlier negative receipts,
including the rejected stale-object run, remain preserved.
