#pragma once

// A decoder weight matrix on the device in the format the engine was asked
// for: bf16 (the checkpoint's own values) or int8 with one fp32 scale per
// 32 inputs of a row, quantised at load time on the host (round to nearest,
// symmetric). Several matrices of the same shape can be stacked into one
// weight so a kernel can pick a row block by index (the routed experts).

#include <cstdint>
#include <string>
#include <vector>

#include "backend/cuda/context.h"
#include "backend/cuda/gemv_ops.h"
#include "model/weight_source.h"
#include "tensor/tensor.h"

namespace uocr::cuda {

enum class WeightFormat : std::uint8_t { kBf16, kInt8 };

struct DeviceWeight {
  WeightFormat format = WeightFormat::kBf16;
  Array data;    // bf16 or int8, [rows, cols]
  Array scales;  // int8 only: fp32 [rows, cols / 32]
  std::int64_t rows = 0;
  std::int64_t cols = 0;

  // The row block starting at `row` (for stacked weights).
  const std::uint16_t* bf16_at(std::int64_t row) const { return data.bf16() + row * cols; }
  Q8Weights q8_at(std::int64_t row) const {
    return Q8Weights{data.i8() + row * cols, scales.f32() + row * (cols / kQ8Group)};
  }
};

// One matrix (trailing dims flattened into cols), or several stacked by rows.
DeviceWeight device_weight(const Context& ctx, const Tensor& t, WeightFormat format);
DeviceWeight stacked_weight(const Context& ctx, const WeightSource& w, const std::vector<std::string>& names,
                            WeightFormat format);

// y[m, rows] = x[m, cols] w^T (+ bias), for a row block of `n` rows at `row`.
void apply_weight(const Context& ctx, const float* x, std::int64_t m, const DeviceWeight& w, const float* bias, float* y,
                  std::int64_t row = 0, std::int64_t n = -1);

// Host-side quantisation, exposed for tests: int8 values and per-group scales.
void quantize_q8_rows(const float* w, std::int64_t rows, std::int64_t cols, std::int8_t* q, float* scales);

}  // namespace uocr::cuda
