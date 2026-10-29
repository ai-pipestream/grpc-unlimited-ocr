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

// Two-axis rope on x [rows, heads * dim], one (tok, page) per row.
void rope(const Context& ctx, float* x, std::int64_t rows, std::int64_t heads, std::int64_t dim, const std::int64_t* tok,
          const std::int64_t* page, const float* inv_tok, std::int64_t n_tok, const float* inv_page, std::int64_t n_page,
          std::int64_t page_mod);

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
// bf16 table rows -> fp32 rows (token embeddings).
void gather_rows_bf16(const Context& ctx, const std::uint16_t* table, const std::int64_t* index, std::int64_t rows,
                      std::int64_t cols, float* dst);

// SAM's decomposed relative-position bias for one head of one window:
// bias[t, ky * w + kx] = q[t] . rh[qy, ky] + q[t] . rw[qx, kx], with q rows
// `ldq` apart and rh [h, h, d], rw [w, w, d].
void rel_pos_bias(const Context& ctx, const float* q, std::int64_t ldq, std::int64_t h, std::int64_t w, std::int64_t d,
                  const float* rh, const float* rw, float* bias);

// Router block summaries: for each 32-key block of keys [s, d], the
// head_cls-weighted softmax average of its keys: [nblk, d].
void block_summaries(const Context& ctx, const float* keys, std::int64_t s, std::int64_t d, const float* head_cls,
                     std::int64_t block_size, float* out);

}  // namespace uocr::cuda
