#include "backend/metal/gemm.h"
#include "backend/reference/ops.h"
#include "backend/reference/parallel.h"
#include "check.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
  if (uocr::metal::Gemm::device_count() == 0) return 77;
  uocr::reference::set_thread_count(2);
  uocr::metal::Gemm gpu;
  std::cout << "device=" << gpu.device_name() << " registry_id=" << gpu.device_id() << '\n';
  for (auto dims : {std::vector<std::int64_t>{1, 7, 13}, {3, 11, 17}, {32, 128, 96}}) {
    const auto m = dims[0], n = dims[1], k = dims[2];
    uocr::Tensor x({m, k}), w({n, k});
    for (std::int64_t i = 0; i < x.numel(); ++i) x.data()[i] = std::sin(static_cast<float>(i));
    for (std::int64_t i = 0; i < w.numel(); ++i) w.data()[i] = std::cos(static_cast<float>(i));
    const auto oracle = uocr::reference::linear(x, w);
    for (int repeat = 0; repeat < 5; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      const auto actual = gpu.linear(x, w);
      require(actual.shape() == oracle.shape(), "Metal linear preserves output shape");
      for (std::int64_t i = 0; i < actual.numel(); ++i)
        require(std::isfinite(actual.data()[i]) &&
          std::fabs(actual.data()[i] - oracle.data()[i]) <= 2e-4f * std::max(1.0f, std::fabs(oracle.data()[i])),
          "Metal fp32 GEMM matches reference within 2e-4 relative/absolute tolerance");
      std::cout << "shape=" << m << ',' << n << ',' << k << " repeat=" << repeat << " us="
        << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count() << '\n';
    }
  }
  bool rejected = false;
  try { uocr::metal::Gemm limited(0, 16); (void)limited.linear(uocr::Tensor({2, 2}), uocr::Tensor({2, 2})); }
  catch (const std::length_error&) { rejected = true; }
  require(rejected, "aggregate device budget enforced before allocation");
  rejected = false;
  try { uocr::metal::Gemm missing(uocr::metal::Gemm::device_count()); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "missing Metal device fails without CPU fallback");
}
