#include "model/intel/device_weight.h"

#include <stdexcept>

#include "backend/intel/native/ops.h"
namespace uocr::intel_native {
DeviceWeight device_weight(const Context& ctx, const Tensor& t, WeightFormat) {
  return {upload_tensor(ctx, t, DType::kBF16), t.dim(0), t.numel() / t.dim(0)};
}
DeviceWeight stacked_weight(const Context& ctx, const WeightSource& w, const std::vector<std::string>& names,
                            WeightFormat) {
  const Tensor t = w.get(names.at(0));
  const auto rows = t.dim(0), cols = t.numel() / rows;
  DeviceWeight dw{Array(ctx, DType::kBF16, t.numel() * static_cast<std::int64_t>(names.size())),
                  rows * static_cast<std::int64_t>(names.size()), cols};
  for (std::size_t i = 0; i < names.size(); ++i) {
    const Tensor v = i == 0 ? t : w.get(names[i]);
    if (v.dim(0) != rows || v.numel() != t.numel()) throw std::invalid_argument("expert matrix shape mismatch");
    dw.data.upload(v.data(), v.numel(), static_cast<std::int64_t>(i) * v.numel());
  }
  return dw;
}
void apply_weight(const Context& ctx, const float* x, std::int64_t m, const DeviceWeight& w, const float* bias,
                  float* y, std::int64_t row, std::int64_t n) {
  if (n < 0) n = w.rows;
  if (row < 0 || n < 0 || row > w.rows || n > w.rows - row) throw std::invalid_argument("weight slice bounds");
  linear(ctx, x, m, w.cols, w.data.bf16() + row * w.cols, n, bias, y);
}
}  // namespace uocr::intel_native
