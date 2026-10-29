#pragma once

// The CUDA backend's op set, over raw device pointers (fp32 activations,
// bf16 weights) so callers can address slices. Every op is enqueued on the
// context's stream; none synchronises. Shapes are row-major, as in the
// reference backend, so each op mirrors its reference/ops.h counterpart.

#include <cstdint>

#include "backend/cuda/context.h"

namespace uocr::cuda {

// y[m, n] = x[m, k] w[n, k]^T (+ bias[n]): x is rounded to bf16 for the
// product, accumulation is fp32.
void linear(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, const std::uint16_t* w_bf16,
            std::int64_t n, const float* bias, float* y);

// Plain fp32 products for attention, batched over `batch` slices that sit
// `stride_*` elements apart (row-major, leading dimensions given):
//   c = alpha * a[m, k] b[n, k]^T          (gemm_nt)
//   c = a[m, k] b[k, n]                    (gemm_nn)
void gemm_nt(const Context& ctx, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
             std::int64_t lda, std::int64_t stride_a, const float* b, std::int64_t ldb, std::int64_t stride_b, float* c,
             std::int64_t ldc, std::int64_t stride_c, std::int64_t batch);
void gemm_nn(const Context& ctx, std::int64_t m, std::int64_t n, std::int64_t k, const float* a, std::int64_t lda,
             std::int64_t stride_a, const float* b, std::int64_t ldb, std::int64_t stride_b, float* c, std::int64_t ldc,
             std::int64_t stride_c, std::int64_t batch);

void add_inplace(const Context& ctx, float* y, const float* x, std::int64_t n);         // y += x
void add_rows(const Context& ctx, float* y, const float* row, std::int64_t rows, std::int64_t cols);  // y[r] += row
void layer_norm(const Context& ctx, float* x, const float* w, const float* b, std::int64_t rows, std::int64_t cols,
                float eps);
void rms_norm(const Context& ctx, float* x, const float* w, std::int64_t rows, std::int64_t cols, float eps);
// The router's norm: scale by (1 + w), non-finite inputs zeroed, clamped.
void router_norm(const Context& ctx, float* x, const float* w, std::int64_t rows, std::int64_t cols, float eps);
void gelu_erf(const Context& ctx, float* x, std::int64_t n);
void quick_gelu(const Context& ctx, float* x, std::int64_t n);
void silu_mul(const Context& ctx, float* g, const float* u, std::int64_t n);  // g = silu(g) * u

// Softmax over each row of x [rows, cols] in place; a fully masked row
// (all -inf) becomes zeros.
void softmax_rows(const Context& ctx, float* x, std::int64_t rows, std::int64_t cols);
// x[b, r, c] = x[b, r, c] * scale + mask[r, c], for each of `batch` slices.
void scale_add_mask(const Context& ctx, float* x, float scale, const float* mask, std::int64_t batch, std::int64_t rows,
                    std::int64_t cols);

// Two-axis rope on x [rows, heads * dim], one (tok, page) per row; with
// null tok/page arrays the single row uses `one_tok` / `one_page`.
void rope(const Context& ctx, float* x, std::int64_t rows, std::int64_t heads, std::int64_t dim, const std::int64_t* tok,
          const std::int64_t* page, const float* inv_tok, std::int64_t n_tok, const float* inv_page, std::int64_t n_page,
          std::int64_t page_mod, std::int64_t one_tok = 0, std::int64_t one_page = 0);

// [C, H, W] -> [OH * OW, C * KH * KW] patches, weight layout order.
void im2col(const Context& ctx, const float* x, std::int64_t c, std::int64_t h, std::int64_t w, std::int64_t kh,
            std::int64_t kw, int stride, int pad, float* cols);
void chw_to_hwc(const Context& ctx, const float* x, std::int64_t c, std::int64_t h, std::int64_t w, float* y);
void hwc_to_chw(const Context& ctx, const float* x, std::int64_t h, std::int64_t w, std::int64_t c, float* y);

// dst[i] = src[index[i]] row-wise; scatter-add dst[index[i]] += weight[i] * src[i].
void gather_rows(const Context& ctx, const float* src, const std::int64_t* index, std::int64_t rows, std::int64_t cols,
                 float* dst);
void scatter_add_rows(const Context& ctx, float* dst, const std::int64_t* index, const float* weight, const float* src,
                      std::int64_t rows, std::int64_t cols);
// out[r] = a[r] (ca columns) followed by b[r] (cb columns).
void concat_cols(const Context& ctx, const float* a, std::int64_t ca, const float* b, std::int64_t cb, std::int64_t rows,
                 float* out);
// dst = src for n elements (device to device).
void copy(const Context& ctx, float* dst, const float* src, std::int64_t n);
// bf16 table rows -> fp32 rows (token embeddings); one_index serves a
// single row without an index array.
void gather_rows_bf16(const Context& ctx, const std::uint16_t* table, const std::int64_t* index, std::int64_t rows,
                      std::int64_t cols, float* dst);
void gather_row_bf16(const Context& ctx, const std::uint16_t* table, std::int64_t one_index, std::int64_t cols,
                     float* dst);

// Mixture-of-experts routing on the device: softmax over each row of gate
// logits [rows, experts], then the top k (lower index on a tie) as expert
// ids and their scores.
void expert_topk(const Context& ctx, const float* logits, std::int64_t rows, std::int64_t experts, std::int64_t k,
                 std::int64_t* ids, float* weights);
// One decode row through its k routed experts, weights stacked by expert:
// gate/up [experts * ffn, hidden], down [experts * hidden, ffn].
//   out[c] = sum_j weight[j] * down[ids[j]] (silu(gate[ids[j]] x) * up[ids[j]] x)
// with x and the hidden activation rounded to bf16 for the products, as
// in `linear`.
void expert_ffn_one(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn,
                    const std::uint16_t* gate, const std::uint16_t* up, const std::uint16_t* down,
                    const std::int64_t* ids, const float* weights, std::int64_t k, float* out);

// SAM's decomposed relative-position bias for one head of one window:
// bias[t, ky * w + kx] = q[t] . rh[qy, ky] + q[t] . rw[qx, kx], with q rows
// `ldq` apart and rh [h, h, d], rw [w, w, d].
void rel_pos_bias(const Context& ctx, const float* q, std::int64_t ldq, std::int64_t h, std::int64_t w, std::int64_t d,
                  const float* rh, const float* rw, float* bias);

// Router block summaries: for each 32-key block of keys [s, d], the
// head_cls-weighted softmax average of its keys: [nblk, d].
void block_summaries(const Context& ctx, const float* keys, std::int64_t s, std::int64_t d, const float* head_cls,
                     std::int64_t block_size, float* out);

// Block routing on the device, the same rule as model/block_router.h's
// select_blocks plus attention_plan.h's key_mask, so no score leaves the
// device. For each query row i at position first_pos + i: block scores z
// [rows, nblk] (already scaled by 1/sqrt(d)) are clamped and limited to the
// blocks before the local window, restricted to the blocks holding image
// tokens of the query's page (page ranges per prompt block in block_min /
// block_max, the query's page in query_page[i] or, when query_page is null,
// `one_page`; a negative page or no such block means no restriction), top
// `top_m` go through 1.5-entmax, the first `top_k` are read with a
// log-prior bias, and the row's key mask [keys] is written.
struct RouteParams {
  std::int64_t block_size;
  std::int64_t local_blocks;
  std::int64_t top_m;
  std::int64_t top_k;
  float idx_scale;
  float floor;
  float sigma;
};
void route_blocks(const Context& ctx, const float* z, std::int64_t rows, std::int64_t nblk, std::int64_t first_pos,
                  std::int64_t keys, const std::int64_t* block_min, const std::int64_t* block_max,
                  std::int64_t prompt_blocks, const std::int64_t* query_page, std::int64_t one_page,
                  const RouteParams& params, float* mask);

}  // namespace uocr::cuda
