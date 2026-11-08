#include "backend/intel/native/context_impl.h"
#include "backend/intel/native/ops.h"
namespace uocr::intel_native {
namespace {
using DT = dnnl::memory::data_type;
void matmul(const Context& ctx, const void* a, const void* b, float* c, const dnnl::memory::desc& ad,
            const dnnl::memory::desc& bd, const dnnl::memory::desc& cd) {
  auto& impl = ctx.impl();
  const auto am =
      dnnl::sycl_interop::make_memory(ad, impl.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(a));
  const auto bm =
      dnnl::sycl_interop::make_memory(bd, impl.engine, dnnl::sycl_interop::memory_kind::usm, const_cast<void*>(b));
  const auto cm = dnnl::sycl_interop::make_memory(cd, impl.engine, dnnl::sycl_interop::memory_kind::usm, c);
  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
  attr.set_deterministic(true);
  auto pd = dnnl::matmul::primitive_desc(impl.engine, ad, bd, cd, attr);
  const auto sd = pd.scratchpad_desc();
  Array scratch(ctx, DType::kI8, static_cast<std::int64_t>(std::max<std::size_t>(sd.get_size(), 1)));
  const auto sm =
      dnnl::sycl_interop::make_memory(sd, impl.engine, dnnl::sycl_interop::memory_kind::usm, scratch.data());
  impl.gemms.fetch_add(1);
  dnnl::matmul(pd).execute(impl.stream,
                           {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm}, {DNNL_ARG_DST, cm}, {DNNL_ARG_SCRATCHPAD, sm}});
}
}  // namespace
void linear(const Context& ctx, const float* x, std::int64_t m, std::int64_t k, const std::uint16_t* w, std::int64_t n,
            const float* bias, float* y) {
  if (m == 1) {
    // Each subgroup owns an output row; no host dot products or full-weight expansion.
    ctx.impl().kernels.fetch_add(1);
    ctx.impl().queue.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(n) * 32, 32),
                                  [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                    const auto row = static_cast<std::int64_t>(it.get_group(0)),
                                               lane = static_cast<std::int64_t>(it.get_local_id(0));
                                    float sum = 0;
                                    for (std::int64_t j = lane; j < k; j += 32)
                                      sum += wide(narrow(x[j])) * wide(w[row * k + j]);
                                    sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
                                    if (lane == 0) y[row] = sum + (bias ? bias[row] : 0);
                                  });
    return;
  }
  Array xb(ctx, DType::kBF16, m * k);
  auto* p = xb.bf16();
  each(ctx, m * k, [=](auto i) { p[i] = narrow(x[i]); });
  matmul(ctx, p, w, y, {{m, k}, DT::bf16, {k, 1}}, {{k, n}, DT::bf16, {1, k}}, {{m, n}, DT::f32, {n, 1}});
  if (bias) add_rows(ctx, y, bias, m, n);
}
void gemm_nt(const Context& c, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float* a,
             std::int64_t lda, std::int64_t sa, const float* b, std::int64_t ldb, std::int64_t sb, float* out,
             std::int64_t ldc, std::int64_t sc, std::int64_t batch) {
  for (std::int64_t i = 0; i < batch; ++i) {
    matmul(c, a + i * sa, b + i * sb, out + i * sc, {{m, k}, DT::f32, {lda, 1}}, {{k, n}, DT::f32, {1, ldb}},
           {{m, n}, DT::f32, {ldc, 1}});
    if (alpha != 1) {
      auto* p = out + i * sc;
      each(c, m * n, [=](auto j) { p[(j / n) * ldc + j % n] *= alpha; });
    }
  }
}
void gemm_nn(const Context& c, std::int64_t m, std::int64_t n, std::int64_t k, const float* a, std::int64_t lda,
             std::int64_t sa, const float* b, std::int64_t ldb, std::int64_t sb, float* out, std::int64_t ldc,
             std::int64_t sc, std::int64_t batch) {
  for (std::int64_t i = 0; i < batch; ++i)
    matmul(c, a + i * sa, b + i * sb, out + i * sc, {{m, k}, DT::f32, {lda, 1}}, {{k, n}, DT::f32, {ldb, 1}},
           {{m, n}, DT::f32, {ldc, 1}});
}
}  // namespace uocr::intel_native
