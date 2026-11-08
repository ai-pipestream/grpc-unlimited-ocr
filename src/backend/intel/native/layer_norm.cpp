#include "backend/intel/native/context_impl.h"
#include "backend/intel/native/ops.h"

namespace uocr::intel_native {
void layer_norm(const Context& ctx, float* x, const float* weight, const float* bias, std::int64_t rows,
                std::int64_t cols, float eps) {
  // This source uses precise FP: reassociating (x-origin)-mean loses the
  // small centered values when the original input has a large offset.
  each(ctx, rows, [=](auto row) {
    float* values = x + row * cols;
    const float origin = values[0];
    float sum = 0;
    for (std::int64_t col = 0; col < cols; ++col) sum += values[col] - origin;
    const float mean = sum / static_cast<float>(cols);
    float variance = 0;
    for (std::int64_t col = 0; col < cols; ++col) {
      const float centered = (values[col] - origin) - mean;
      variance += centered * centered;
    }
    const float inverse = sycl::rsqrt(variance / static_cast<float>(cols) + eps);
    for (std::int64_t col = 0; col < cols; ++col)
      values[col] = ((values[col] - origin) - mean) * inverse * weight[col] + bias[col];
  });
}
}  // namespace uocr::intel_native
