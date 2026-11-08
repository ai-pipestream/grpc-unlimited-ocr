# Metal performance audit and next qualification plan

This is an offline source audit against native implementation
`410b20f61ff2e35c08113dc1fde9a509dd4667a6`, using the existing M2 receipts.
No Mac commands, builds or GPU jobs were issued for this audit. Kant owns
that machine's current slot. No optimization speedup is claimed here.
The reported CUDA time on another machine is not a valid speed ratio.

## Existing measurements and attribution limits

The warm real page took 43.981 s: 18.420 s vision/prefill and 25.561 s
decode. It issued 71,903 native compute dispatches and 53,264 MPS GEMM
encodes. Retained owned buffers were 6,829,784,464 bytes; peak owned bytes
were 7,315,618,256. These are measured counters, not a per-kernel GPU
profile. There are no recorded command-buffer wait times, allocation
counts or CPU encode times. The earlier receipt notes possible compilation
overlap. The CPU baseline also experienced heavy paging. Neither supports
an isolated cross-device performance conclusion.

## Source-derived costs

The following counts are predictions for the recorded 171-token page,
with default dimensions and no cache growth or allocation-pressure flush.
The generator makes 171 single-row decoder calls: prefill predicts the
first token, and the final decoder call predicts EOS. These predictions
must be checked against instrumentation before attributing elapsed time.

- Each decoder step invokes 263 MPS encodes: 12 layers times two
  ten-head attention products plus one block-router product, and 11 MoE
  gate products. That is 44,973 encodes, 84.43% of the measured whole-page
  53,264. All have M=1. Existing bf16 linear GEMV already avoids MPS;
  the remaining products use fp32 inputs.
- `product()` creates one `MPSMatrixMultiplication` per call, but loops
  over heads and creates three descriptors/matrix wrappers per encoded
  product. Decode therefore creates 8,037 multiplication objects and
  134,919 matrix wrappers. This is source-derived object churn, not a
  measured allocation-time share.
- `step()`/`run()` and their helpers allocate 156 temporary Arrays per
  step, or 26,676 new shared MTLBuffers per page, excluding cache growth.
  Each uses `newBufferWithLength`, an ordered-map insertion and eventual
  retirement. Existing deferred retirement prevents premature reuse but
  does not pool storage.
- Each step issues 360 native kernels and 47 `product()` calls, hence
  407 increments of the pending counter. The 64 threshold predicts six
  intermediate submit/waits and the final logits download wait: seven
  waits per token, 1,197 for decode. A ten-head product increments pending
  only once although it encodes ten GEMMs: the threshold is not literally
  64 GPU operations. Additional cap-pressure flushes would increase this.
- Prefill calls `synchronize()` after top-k, then uploads expert grouping
  IDs and probabilities into new buffers. Every `Array::upload()` first
  drains the queue even for freshly allocated storage. For each active
  expert, the ID upload can serialize the preceding expert's work; the
  immediately following weight upload often finds no pending commands.
  The number of active experts is input dependent and was not logged.
- Single-row norms execute one GPU thread over 1,280 elements. Softmax
  similarly executes one thread per row. Block summaries recompute every
  block each decode step, even though only the last block has changed.
  These are low-parallelism kernel candidates, not measured hotspots.

See `metal-m2-performance-20261108/source-counts.txt` for the arithmetic.

## First change: fresh-buffer upload without an unrelated queue drain

Add a constructor/factory that allocates an Array and initializes its
shared bytes before exposing it to any encoder. Keep ordinary `upload()`
synchronizing: it must remain safe for overwrites of buffers with pending
users. Use the fresh path for `ids_on_device`, `upload_tensor`, newly
created prefill grouping weights and immutable position-table uploads.
Do not relax the existing allocation cap or retired-buffer accounting.

Why this is a narrow first change: no reduction order, model precision,
operator shape or weight representation changes. The new buffer has never
been encoded, and pending retired storage cannot be recycled before its
users finish. A cap-pressure drain in allocation still remains valid.
Do not simply remove `synchronize()` from the general upload method.

Gate before a model run:

1. Expose command submission counters and show that fresh upload does not
   increase them while an unrelated copy is pending. After the eventual
   synchronize, assert both the older copy and fresh contents exactly.
2. Overwrite a previously encoded buffer through ordinary `upload()` and
   verify that it still drains before modification. Retain the tiny-cap,
   deferred-lifetime and injected sticky-error tests.
3. Run existing ops, tiled vision and 44-step decoder parity without
   relaxing tolerances; require real golden text/labels/geometry and
   repeat tokens, retained memory and three TCP recovery cycles unchanged.
4. A/B prefill on the same M2, same head and exact source pins, same page,
   no overlapping models/builds, alternating baseline/candidate order.
   Log grouping-upload drain count/time and total vision/prefill separately.

## Next change: native fp32 single-row products

Specialize `product()` for M=1. Keep MPS for multirow GEMM. For NT, use a
SIMD reduction of each contiguous B row, respecting lda/ldb and head batch
strides. For NN, map adjacent lanes to adjacent output columns so V reads
are coalesced; preserve its strided output layout. Batch all heads in one
native dispatch, rather than ten encodes. Block-router and MoE gate
products can use the same fp32 NT path. Do not repurpose bf16 GEMV by
rounding activations or cached keys/values.

Potential structural saving: eliminate 44,973 decode MPS encodes and their
wrappers in this fixture. Replacement native dispatches and arithmetic
still cost time; there is no predicted wall-clock speedup yet. Validate
nonunit alpha, K/N tails, nonzero head offsets, interleaved row strides,
single/multiple heads and sentinel gaps in output. Compare against both
reference products and the existing MPS path. Reduction-order differences
must pass existing decoder and real-page gates, especially routing ties.
Keep this separate from fusion so regressions have one numerical cause.

Then consider fused decode QK + mask + softmax + AV with fp32 accumulation
and exact all-masked-row semantics. It could eliminate scores storage and
additional launches, but masked/page-boundary and long-context cases make
it a larger second-stage change. SIMD norms and fused copy+RMSNorm are
also candidates; reference uses higher-precision reductions, so golden
and near-tie router parity matter. Preserve upstream bf16 weights.

## Allocation and submission work

Prefer persistent decoder scratch for fixed-size q/rq, hidden/delta,
expert intermediates and logits before a general buffer pool. Variable
scores/masks can grow geometrically with KV capacity. Reuse is safe only
after the prior encoded users, within the same ordered command stream;
public request cancellation and subsequent recovery must not expose stale
contents. Count reserved capacity against the cap and test retained bytes
across repeated requests of different lengths.

Do not raise 64 blindly. It currently combines submit with a blocking
wait, but larger batches retain more temporary storage and can worsen
cancellation latency. After scratch reuse and counters, compare 64/128/256
under the same cap and golden/cancellation gates. An asynchronous bounded
in-flight queue requires per-submission retirement and explicit failure
propagation; the current single-command retirement vector cannot simply
be cleared at submit. MPS multiplication-object caching by complete
shape/transpose/alpha key is a lower-numerical-risk control experiment,
but it retains head-by-head encoding and matrix wrappers.

Next profiling instrumentation should be opt-in and bounded: phase
snapshots of native/MPS counts; command submit count, flush reason and
host wait duration; CPU encode duration; new-buffer count/bytes and peak
retirement; shape histograms with a fixed upper bound. GPU start/end
intervals can be sampled at existing completion points, never by adding
a synchronize per operator. Keep traced timing separate from the untraced
A/B run. Disable observers for the final timing unless their overhead is
measured. No profiling or qualification starts until parent/Kant release.

## Combined-branch compatibility audit

Read-only snapshot of parent `feat/native-accelerators`:
`2564252661e14d9a6f5f7fc98fedf3fa78a8765b`. The branch was moving, so this
finding is pinned to that snapshot, not a later head. Parent files were
not edited; sources were extracted with `git archive` into `/tmp`.

The Metal backend, model, context test and `cmake/metal.cmake` match our
qualified implementation. Intel target creation, SYCL compiler probing,
oneDNN discovery, model includes, construction and ownership are gated by
`UOCR_WITH_INTEL`. With Intel OFF, none of those dependency requirements
are introduced into the Metal build. Intel target additions after the
server include still apply target definitions at generation time. Context
members continue to outlive model objects.

Offline GCC C++23 WERROR syntax checks passed with Metal=1, Intel=0,
CUDA=0 for the combined runner, model engine, Metal vision/decoder and
weight adapter. The exact stb003501 headers were used for the engine
check. `cmake/server.cmake` is unchanged from our native-qualified version,
including installed gRPC support. No blocking Metal build-compatibility
issue was found in these additions. The config cap comment mentions only
Intel although Metal also honors it; that is documentation, not a build
failure. This is not an AppleClang/Objective-C++ link qualification of the
combined tree, nor support for enabling Intel SYCL and Metal together.
A native configure/build of the pinned combined tree remains the final
integration gate when the slot is released.
