# M2 requalification after independent correctness review

This receipt supersedes the earlier golden/RPC receipt for the three
failure paths in `/tmp/uocr-independent-review.txt`. Performance work is
frozen in its separate worktree; no performance code is in this build.

## Tested changes

- `e8d238457e8fa6237e18da8b8958a50279b21497`: resolve all buffer bindings
  before opening a compute encoder, reject a nil encoder, and close the
  encoder through a scope guard. Native tests reject cross-context and
  one-past-end pointers, recover with a valid dispatch, and exercise clean
  destruction with older work still pending.
- `d7e2d95e590851a81df8f010d0ae7248d7b1eacd`: LayerNorm uses a shifted mean
  and centered second variance pass. RMSNorm and router normalization keep
  their definitions. Finite rows near 10000, including constant rows and
  low-variance rows, compare against the reference at widths 2/768/1024.
- `d0baa6bc5cf11d3ae59dfea3d3b040d913e92c67`: Metal implements the parent's
  noexcept reset interface. It drains pending work before releasing all
  request cache arrays and block metadata, resets capacity/length/page
  bookkeeping, and is idempotent. Model weights remain resident. A sticky
  device execution failure is not cleared or disguised as recovery.
- Parent common `151443f71bc86dbee089a6e6d05ba40dba3abff8` is imported as
  `27f9ec8495775092ac9c16efb57ff6967f96e9b9`; no separate edits were made to
  its interface, generation exception guards or common recovery test.

## Native failure-path results

Apple M2, native arm64, macOS 27.2, AppleClang 21, WERROR enabled. The
service, model tests and actual RPC harness rebuilt successfully. With
`MTL_DEBUG_LAYER=1`, all six tests passed in 1.74 seconds:

- context lifecycle and invalid-binding recovery;
- native decoder allocation failure/reset/reuse;
- numerical ops including offset LayerNorm;
- tiled vision parity;
- decoder prefill and 44-step parity;
- common generator exception cleanup.

The native decoder recovery test confirms these distinct cases:

1. One-shot allocation failure after exactly the first K-cache replacement
   during growth, with capacity still uncommitted.
2. A real aggregate cap failure at the same partial-growth point, without
   injection, followed by a successful small request under that same cap.
3. Scratch allocation failure after all reserve allocations succeed.
4. A native allocation failure through the real shared generation guard,
   followed by a successful small request on the same decoder.

Each reset returns owned bytes to the model-weight baseline, clears all
request metadata and drains pending/retired users. Recovered logits pass
unchanged 2e-2 reference tolerance. The allocation injection field is
internal, one-shot, disabled by default and has no service/environment
control. The small native fixture, not the full checkpoint, exercises
these failures deterministically.

Offset LayerNorm maximum relative error: width 2, 5.96e-8; widths 768 and
1024, 7.80e-8. The gate remains 1e-5. All Metal tests logged API validation
enabled. Raw logs and source manifest are in
`metal-m2-correctness-20261108/`. The common generator test also passed on
Linux in 0.01 seconds; that is host regression evidence only.

## Source identity

The per-file manifest was checked on the Mac before the full-model runs.
Actual stb is `003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5`; JSON is
`55f93686c01528224f448c19128836e7df245f72` (v3.12.0); both source overrides
are empty. gRPC is explicitly the installed native 1.85.1 package with
`UOCR_SYSTEM_GRPC=ON`. Upstream model files are unchanged from the earlier
SHA256-verified copy. No inference framework or converted weights.

## Full-model revalidation

Fresh two-repeat golden passed with API validation enabled, nice 10,
a 900-second bound and the unchanged 12 GiB owned Metal buffer cap.
Both repeats produced identical 171-token output, exact text/labels and
six boxes with zero coordinate difference. Both finished with EOS.
Retained owned bytes were 6,829,784,464 on both runs; device allocation
was 6,830,735,360 on both; owned peak was 7,315,618,256. The process's
reported peak RSS was 6,604,357,632 bytes. Retention assertions passed.

Total test time was 72.59 seconds; generation observations were 25.458 and
24.642 seconds. These correctness runs are not controlled performance
experiments, and no improvement is attributed to these fixes. The final
external `time` footprint in `golden.txt` belongs to the CTest parent;
the in-test getrusage/Metal counters above describe the model process.

All three subsequent cancellation/recovery cycles passed over actual
127.0.0.1 TCP gRPC using `server_model_stream_test --cancel-only`, with
Metal/bf16 selected explicitly. Each cycle sends a valid raster, waits
for prefill, cancels, checks CANCELLED status and admission release, then
validates a subsequent request's text, boxes, tokens, fragments, completion
and status. Final leases were released. Total wall time was 113.41 seconds,
peak RSS 8,470,855,680 bytes and peak process footprint 7,441,590,544 bytes.
Client and server share a process but communicate through a real TCP
connection. This targeted run covers three one-page recovery requests;
it does not rerun the earlier six alternating one/two-page requests.

Both full-model processes exited successfully with API validation enabled.
Before the sequence no other model remained and memory was 79% free;
afterward memory was 73% free. The Mac slot was released through the
parent/Kant status file. No performance job was run or resumed. These
are fresh post-fix receipts, not substituted earlier results.
