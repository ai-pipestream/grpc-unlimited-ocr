# Metal primitive implementation checkpoint

`UOCR_WITH_METAL=ON` builds a native Objective-C++ bridge to Metal and
MPSMatrixMultiplication. MPS is used only for dense GEMM. The forward pass
has not been replaced with an inference runtime. The primitive implements
fp32 x times transposed weight, with explicit host transfers, a 256 MiB
aggregate device-buffer cap and one serialized command at a time.

The standalone `metal_gemm_test` compares against the unchanged reference
backend with 2e-4 relative/absolute tolerance over three shapes and five
repetitions per shape. It also checks a missing device and an aggregate
allocation overflow. It exits 77 without a device. A skipped test is not
qualification. Only its C++ syntax and the non-Apple configure rejection
have been checked on Linux; the Objective-C++ implementation is uncompiled.

This is not a usable Metal service backend. Native vision, decoder,
attention, normalization, activation, routing, quantized weights and
device-resident model integration remain missing. The service still rejects
Metal. No CPU execution is disguised as Metal.

Actual-device testing is blocked: on 2026-11-08 Mac DNS failed and SSH to
the verified Tailscale peer 100.102.192.43 timed out. No chip/RAM inventory,
Mac model load, golden output, RPC, cancellation/recovery or warm timing
comparison has run. Do not integrate this checkpoint as completed support.
