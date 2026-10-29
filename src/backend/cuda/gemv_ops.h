#pragma once

// Weight-streaming kernels: products where one input row meets a whole
// weight matrix, so the time is the time to read the weights. bf16
// weights, and int8 weights with one fp32 scale per 32 consecutive inputs
// of a row (the format quantize_q8 writes). In both, the input row is
// rounded to bf16 first, as linear() does, so every format sees the same
// activations.

#include <cstdint>

#include "backend/cuda/context.h"

namespace uocr::cuda {

inline constexpr std::int64_t kGemvMaxK = 12288;
inline constexpr std::int64_t kQ8Group = 32;

struct Q8Weights {
  const std::int8_t* q;  // [rows, k]
  const float* scales;   // [rows, k / kQ8Group]
};

void gemv_bf16(const Context& ctx, const float* x, std::int64_t k, const std::uint16_t* w, std::int64_t n,
               const float* bias, float* y);

// y[m, n] = x[m, k] w^T: one row through a GEMV, more rows by expanding the
// weights to bf16 in scratch and using the bf16 GEMM.
void linear_q8(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, Q8Weights w, std::int64_t n,
               const float* bias, float* y);

// expert_ffn_one (ops.h) with int8 expert stacks.
void expert_ffn_one_q8(const Context& ctx, const float* x, std::int64_t hidden, std::int64_t ffn, Q8Weights gate,
                       Q8Weights up, Q8Weights down, const std::int64_t* ids, const float* weights, std::int64_t k,
                       float* out);

}  // namespace uocr::cuda
