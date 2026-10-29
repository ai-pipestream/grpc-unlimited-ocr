#include "model/cuda/device_weight.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "backend/cuda/ops.h"

namespace uocr::cuda {

void quantize_q8_rows(const float* w, std::int64_t rows, std::int64_t cols, std::int8_t* q, float* scales) {
  if (cols % kQ8Group != 0) throw std::invalid_argument("quantize_q8_rows: width must be a multiple of 32");
  const std::int64_t groups = cols / kQ8Group;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t g = 0; g < groups; ++g) {
      const float* src = w + r * cols + g * kQ8Group;
      float peak = 0.0f;
      for (std::int64_t i = 0; i < kQ8Group; ++i) peak = std::max(peak, std::fabs(src[i]));
      const float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
      scales[r * groups + g] = scale;
      for (std::int64_t i = 0; i < kQ8Group; ++i) {
        q[r * cols + g * kQ8Group + i] = static_cast<std::int8_t>(std::clamp(std::nearbyint(src[i] / scale), -127.0f, 127.0f));
      }
    }
  }
}

namespace {

void fill(const Context& ctx, DeviceWeight& dw, const Tensor& t, std::int64_t row_offset) {
  const std::int64_t rows = t.dim(0), cols = t.numel() / t.dim(0);
  if (cols != dw.cols) throw std::invalid_argument("stacked weights must share a width");
  if (dw.format == WeightFormat::kBf16) {
    dw.data.upload(t.data(), t.numel(), row_offset * cols);
    return;
  }
  std::vector<std::int8_t> q(static_cast<std::size_t>(t.numel()));
  std::vector<float> s(static_cast<std::size_t>(rows * (cols / kQ8Group)));
  quantize_q8_rows(t.data(), rows, cols, q.data(), s.data());
  dw.data.upload(q.data(), t.numel(), row_offset * cols);
  dw.scales.upload(s.data(), static_cast<std::int64_t>(s.size()), row_offset * (cols / kQ8Group));
  (void)ctx;
}

DeviceWeight allocate(const Context& ctx, std::int64_t rows, std::int64_t cols, WeightFormat format) {
  DeviceWeight dw;
  dw.format = format;
  dw.rows = rows;
  dw.cols = cols;
  dw.data = Array(ctx, format == WeightFormat::kBf16 ? DType::kBF16 : DType::kI8, rows * cols);
  if (format == WeightFormat::kInt8) dw.scales = Array(ctx, DType::kF32, rows * (cols / kQ8Group));
  return dw;
}

}  // namespace

DeviceWeight device_weight(const Context& ctx, const Tensor& t, WeightFormat format) {
  DeviceWeight dw = allocate(ctx, t.dim(0), t.numel() / t.dim(0), format);
  fill(ctx, dw, t, 0);
  return dw;
}

DeviceWeight stacked_weight(const Context& ctx, const WeightSource& w, const std::vector<std::string>& names,
                            WeightFormat format) {
  const Tensor first = w.get(names.front());
  const std::int64_t rows = first.dim(0), cols = first.numel() / first.dim(0);
  DeviceWeight dw = allocate(ctx, rows * static_cast<std::int64_t>(names.size()), cols, format);
  for (std::size_t i = 0; i < names.size(); ++i) {
    fill(ctx, dw, i == 0 ? first : w.get(names[i]), static_cast<std::int64_t>(i) * rows);
  }
  return dw;
}

void apply_weight(const Context& ctx, const float* x, std::int64_t m, const DeviceWeight& w, const float* bias, float* y,
                  std::int64_t row, std::int64_t n) {
  if (n < 0) n = w.rows;
  if (w.format == WeightFormat::kBf16) {
    linear(ctx, x, m, w.cols, w.bf16_at(row), n, bias, y);
  } else {
    linear_q8(ctx, x, m, w.cols, w.q8_at(row), n, bias, y);
  }
}

}  // namespace uocr::cuda
