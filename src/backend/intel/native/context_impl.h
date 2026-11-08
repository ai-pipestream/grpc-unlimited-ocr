#pragma once
#include <atomic>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include "backend/intel/native/context.h"
namespace uocr::intel_native {
struct Context::Impl {
  std::atomic<std::uint64_t> allocated{0}, peak{0};
  std::atomic<std::uint64_t> kernels{0}, gemms{0};
  std::uint64_t cap = 0;
  sycl::queue queue;
  dnnl::engine engine;
  dnnl::stream stream;
  explicit Impl(sycl::device device)
      : queue(
            device,
            [](sycl::exception_list errors) {
              for (const auto& error : errors) std::rethrow_exception(error);
            },
            sycl::property::queue::in_order{}),
        engine(dnnl::sycl_interop::make_engine(device, queue.get_context())),
        stream(dnnl::sycl_interop::make_stream(engine, queue)) {}
};
template <class F>
void each(const Context& ctx, std::int64_t n, F f) {
  ctx.impl().kernels.fetch_add(1);
  ctx.impl().queue.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)),
                                [=](sycl::id<1> i) { f(static_cast<std::int64_t>(i[0])); });
}
inline float wide(std::uint16_t b) { return sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }
inline std::uint16_t narrow(float f) {
  auto b = sycl::bit_cast<std::uint32_t>(f);
  if ((b & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((b >> 16) | 0x40U);
  return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
}  // namespace uocr::intel_native
