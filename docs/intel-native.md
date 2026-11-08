# Native Intel model checkpoint

2026-11-08: native model implemented; RPC qualification and controlled
performance tests remain open. This branch is not ready for release.

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
Full golden repeats and RPC qualification of this precision fix remain open.

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
and recovery still need the real RPC test.

## Actual results

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
runs recorded 7.67/7.73 decode tokens/s. Final performance work remains open.

The common RPC test includes d978a13a4dea8dd23ee8a7d5c4efb99323f1ed11:
6 repeated requests and 3 cancellation/recovery cycles through loopback TCP.
`intel_native_rpc_test` adds full golden content and geometry checks for
every page, exact token/text/block repeat identity and process memory checks.
Neither RPC test has run on Intel at this checkpoint. The isolated service
build continues.

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
