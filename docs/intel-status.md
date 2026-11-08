# Intel native backend: original primitive stage

For the full model implementation and current hardware results, see
[intel-native.md](intel-native.md). This page describes the earlier prototype.

The initial `src/backend/intel` implementation runs fp32 device kernels on
an explicitly selected Intel Level Zero GPU. It owns allocation, queue
ordering, activation, RMS norm, softmax, and a correctness-first dense linear
kernel. It uses no inference runtime or vendor operations. The model and
service are not wired to this layer. Selecting an Intel model backend is
still unsupported; this change does not establish OCR support.

On krick-1, oneAPI compiler 2025.3.3 and Arc Pro B70 driver 1.15.39122 passed
`intel_ops_test`. The checked-in receipt includes absolute errors against the
existing reference backend and a 100-launch tiny linear timing. That timing
is not model throughput or a CPU speedup claim. The test covers odd widths,
fully masked softmax rows, bias/no-bias linear, three activations, RMS norm,
and invalid row/device rejection. Require hardware with `UOCR_REQUIRE_INTEL=1`.

Reproduce without service/dependency downloads from this worktree:

```sh
source /opt/intel/oneapi/setvars.sh
mkdir -p build-intel
icpx -fsycl -std=c++23 -O2 -ffp-contract=off -Isrc \
  src/backend/intel/ops.cpp src/backend/reference/ops.cpp \
  tests/intel_ops_test.cpp -pthread -o build-intel/intel_ops_test
ONEAPI_DEVICE_SELECTOR=level_zero:gpu UOCR_REQUIRE_INTEL=1 \
  build-intel/intel_ops_test
```

The full project also registers this test with `-DUOCR_WITH_INTEL=ON` and a
SYCL compiler. Remaining work: bf16, tuned dense GEMM, layer norm,
convolution, vision attention, decoder sparse attention/RoPE/MoE, model
forward integration, placement metadata, real model RPC text and boxes,
cancellation/recovery, and comparable warm CPU/GPU latency. None of those
acceptance gates is implied by primitive parity.
