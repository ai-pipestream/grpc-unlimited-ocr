#include "backend/intel/ops.h"
#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uocr::intel {
namespace {
sycl::device select(unsigned index) {
  std::vector<sycl::device> devices;
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
      if (device.get_info<sycl::info::device::vendor_id>() == 0x8086) devices.push_back(device);
    }
  }
  if (index >= devices.size()) throw std::runtime_error("requested Intel Level Zero GPU unavailable");
  return devices[index];
}
void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
void same_context(const Array& a, const Array& b) {
  require(&a.context() == &b.context(), "Intel arrays belong to different contexts");
}
void rows(const Array& x, std::size_t width) {
  require(width && x.count() && x.count() % width == 0, "invalid Intel row dimensions");
}
}
struct Context::Impl {
  sycl::queue queue;
  explicit Impl(unsigned index) : queue(select(index), [](sycl::exception_list errors) {
    for (const auto& error : errors) std::rethrow_exception(error);
  }, sycl::property::queue::in_order{}) {}
};
Context::Context(unsigned device) : impl_(std::make_unique<Impl>(device)) {}
Context::~Context() = default;
std::string Context::device_name() const {
  return impl_->queue.get_device().get_info<sycl::info::device::name>();
}
std::string Context::driver_version() const {
  return impl_->queue.get_device().get_info<sycl::info::device::driver_version>();
}
void Context::synchronize() const { impl_->queue.wait_and_throw(); }
Array::Array(const Context& ctx, std::size_t count) : context_(ctx), count_(count), data_(nullptr) {
  require(count && count <= std::numeric_limits<std::size_t>::max() / sizeof(float), "invalid Intel allocation size");
  data_ = sycl::malloc_device<float>(count, ctx.impl().queue);
  if (!data_) throw std::bad_alloc();
}
Array::~Array() {
  // An array can leave scope while kernels still reference it. No throwing
  // destructor; explicit synchronize/download is the asynchronous error gate.
  context_.impl().queue.wait();
  sycl::free(data_, context_.impl().queue);
}
void Array::upload(std::span<const float> values) {
  require(values.size() == count_, "Intel upload size mismatch");
  context_.impl().queue.memcpy(data_, values.data(), count_ * sizeof(float)).wait_and_throw();
}
std::vector<float> Array::download() const {
  std::vector<float> values(count_);
  context_.impl().queue.memcpy(values.data(), data_, count_ * sizeof(float)).wait_and_throw();
  return values;
}
void activate(Array& x, Activation kind) {
  auto* p = x.data();
  x.context().impl().queue.parallel_for(sycl::range<1>(x.count()), [=](sycl::id<1> index) {
    const float v = p[index];
    if (kind == Activation::gelu_erf) p[index] = 0.5f * v * (1.0f + sycl::erf(v * 0.7071067811865475f));
    else p[index] = v / (1.0f + sycl::exp(-(kind == Activation::silu ? 1.0f : 1.702f) * v));
  });
}
void rms_norm(Array& x, const Array& weight, std::size_t width, float epsilon) {
  rows(x, width); same_context(x, weight);
  require(weight.count() == width && std::isfinite(epsilon) && epsilon >= 0, "invalid Intel RMS norm arguments");
  auto* p = x.data(); auto* w = weight.data();
  x.context().impl().queue.parallel_for(sycl::range<1>(x.count() / width), [=](sycl::id<1> row) {
    auto* v = p + row[0] * width;
    float sum = 0;
    for (std::size_t j = 0; j < width; ++j) sum += v[j] * v[j];
    const float inv = sycl::rsqrt(sum / static_cast<float>(width) + epsilon);
    for (std::size_t j = 0; j < width; ++j) v[j] = w[j] * (v[j] * inv);
  });
}
void softmax_rows(Array& x, std::size_t width) {
  rows(x, width); auto* p = x.data();
  x.context().impl().queue.parallel_for(sycl::range<1>(x.count() / width), [=](sycl::id<1> row) {
    auto* v = p + row[0] * width;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < width; ++j) maximum = sycl::fmax(maximum, v[j]);
    if (maximum == -std::numeric_limits<float>::infinity()) {
      for (std::size_t j = 0; j < width; ++j) v[j] = 0;
      return;
    }
    float sum = 0;
    for (std::size_t j = 0; j < width; ++j) { v[j] = sycl::exp(v[j] - maximum); sum += v[j]; }
    for (std::size_t j = 0; j < width; ++j) v[j] /= sum;
  });
}
void linear(const Array& x, const Array& weight, const Array* bias, Array& output,
            std::size_t m, std::size_t n, std::size_t k) {
  same_context(x, weight); same_context(x, output);
  require(m && n && k && x.count() / k == m && x.count() % k == 0 &&
          weight.count() / k == n && weight.count() % k == 0 &&
          output.count() / n == m && output.count() % n == 0, "invalid Intel linear dimensions");
  require(&output != &x && &output != &weight && &output != bias, "Intel linear output must not alias inputs");
  if (bias) { same_context(x, *bias); require(bias->count() == n, "invalid Intel linear bias"); }
  const auto* a = x.data(); const auto* w = weight.data(); const auto* b = bias ? bias->data() : nullptr;
  auto* y = output.data();
  x.context().impl().queue.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> index) {
    float sum = 0;
    for (std::size_t j = 0; j < k; ++j) sum += a[index[0] * k + j] * w[index[1] * k + j];
    y[index[0] * n + index[1]] = sum + (b ? b[index[1]] : 0.0f);
  });
}
}  // namespace uocr::intel
