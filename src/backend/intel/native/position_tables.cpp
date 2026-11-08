#include "backend/intel/native/position_tables.h"

#include <algorithm>
#include <cmath>

#include "backend/intel/native/context_impl.h"
#include "backend/intel/native/ops.h"
namespace uocr::intel_native {
namespace {
float cubic(float x) {
  x = std::fabs(x);
  if (x < 1) return (1.5f * x - 2.5f) * x * x + 1;
  if (x < 2) return ((-0.5f * x + 2.5f) * x - 4) * x + 2;
  return 0;
}
std::vector<float> taps(std::int64_t in, std::int64_t out) {
  const float scale = static_cast<float>(in) / static_cast<float>(out), support = scale >= 1 ? 2 * scale : 2,
              inv = scale >= 1 ? 1 / scale : 1;
  std::vector<float> result(static_cast<std::size_t>(in * out), 0);
  for (std::int64_t i = 0; i < out; ++i) {
    const float center = scale * (static_cast<float>(i) + 0.5f);
    const auto lo = std::max<std::int64_t>(static_cast<std::int64_t>(center - support + 0.5f), 0),
               hi = std::min<std::int64_t>(static_cast<std::int64_t>(center + support + 0.5f), in);
    float total = 0;
    for (auto j = lo; j < hi; ++j) {
      const auto v = cubic((static_cast<float>(j) - center + 0.5f) * inv);
      result[static_cast<std::size_t>(i * in + j)] = v;
      total += v;
    }
    if (total != 0)
      for (auto j = lo; j < hi; ++j) result[static_cast<std::size_t>(i * in + j)] /= total;
  }
  return result;
}
}  // namespace
Array resize_position_grid(const Context& ctx, const Tensor& t, std::int64_t src, std::int64_t dst, std::int64_t width,
                           bool cls) {
  Array input(ctx, DType::kF32, t.numel());
  input.upload(t.data(), t.numel());
  if (src == dst) return input;
  const auto weights = taps(src, dst);
  Array coeff(ctx, DType::kF32, src * dst);
  coeff.upload(weights.data(), src * dst);
  const std::int64_t off = cls ? width : 0;
  Array output(ctx, DType::kF32, dst * dst * width + off);
  auto* out = output.f32();
  const auto* in = input.f32();
  const auto* w = coeff.f32();
  if (cls) copy(ctx, out, in, width);
  each(ctx, dst * dst * width, [=](auto i) {
    const auto oy = i / (dst * width), ox = (i / width) % dst, ch = i % width;
    float acc = 0;
    for (std::int64_t y = 0; y < src; ++y) {
      if (w[oy * src + y] == 0) continue;
      float line = 0;
      for (std::int64_t x = 0; x < src; ++x)
        if (w[ox * src + x] != 0) line += in[off + (y * src + x) * width + ch] * w[ox * src + x];
      acc += line * w[oy * src + y];
    }
    out[off + i] = acc;
  });
  return output;
}
Array relative_position_table(const Context& ctx, const Tensor& table, std::int64_t size) {
  const auto len = table.dim(0), d = table.dim(1), span = 2 * size - 1;
  Array input(ctx, DType::kF32, table.numel());
  input.upload(table.data(), table.numel());
  Array output(ctx, DType::kF32, size * size * d);
  const auto* in = input.f32();
  auto* out = output.f32();
  each(ctx, size * size * d, [=](auto i) {
    const auto q = i / (size * d), k = (i / d) % size, e = i % d, row = q - k + size - 1;
    if (len == span) {
      out[i] = in[row * d + e];
      return;
    }
    const float src =
        sycl::fmax(static_cast<float>(len) / static_cast<float>(span) * (static_cast<float>(row) + 0.5f) - 0.5f, 0.0f);
    const auto i0 = static_cast<std::int64_t>(src), i1 = i0 + (i0 < len - 1 ? 1 : 0);
    const float l1 = src - static_cast<float>(i0);
    out[i] = (1 - l1) * in[i0 * d + e] + l1 * in[i1 * d + e];
  });
  return output;
}
}  // namespace uocr::intel_native
