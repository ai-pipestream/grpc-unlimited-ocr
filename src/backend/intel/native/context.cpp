#include <limits>
#include <stdexcept>
#include <utility>

#include "backend/intel/native/context_impl.h"
#include "tensor/dtype.h"
namespace uocr::intel_native {
namespace {
std::vector<sycl::device> devices() {
  std::vector<sycl::device> out;
  for (const auto& p : sycl::platform::get_platforms()) {
    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
    for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
      if (d.get_info<sycl::info::device::vendor_id>() == 0x8086) out.push_back(d);
  }
  return out;
}
sycl::device selected(int index) {
  auto ds = devices();
  if (index < 0 || static_cast<std::size_t>(index) >= ds.size())
    throw std::runtime_error("requested native Intel Level Zero GPU unavailable");
  return ds[static_cast<std::size_t>(index)];
}
std::size_t size_of(DType t) {
  switch (t) {
    case DType::kF32:
      return 4;
    case DType::kBF16:
      return 2;
    case DType::kI64:
      return 8;
    case DType::kI8:
      return 1;
  }
  throw std::invalid_argument("invalid device dtype");
}
void bounds(std::int64_t size, std::int64_t count, std::int64_t offset) {
  if (offset < 0 || count < 0 || offset > size || count > size - offset)
    throw std::invalid_argument("device upload bounds");
}
}  // namespace
int device_count() { return static_cast<int>(devices().size()); }
DeviceInfo device_info(int index) {
  auto d = selected(index);
  return {d.get_info<sycl::info::device::name>(), d.get_info<sycl::info::device::global_mem_size>()};
}
Context::Context(int index, std::uint64_t cap) : impl_(std::make_unique<Impl>(selected(index))) {
  const auto total = impl_->queue.get_device().get_info<sycl::info::device::global_mem_size>();
  impl_->cap = cap ? cap : total / 5 * 4;
  if (impl_->cap > total) throw std::invalid_argument("Intel memory cap exceeds device capacity");
}
std::uint64_t Context::allocated_bytes() const { return impl_->allocated.load(); }
std::uint64_t Context::peak_bytes() const { return impl_->peak.load(); }
std::uint64_t Context::memory_cap() const { return impl_->cap; }
Context::~Context() = default;
std::uint64_t Context::kernel_submissions() const { return impl_->kernels.load(); }
std::uint64_t Context::gemm_submissions() const { return impl_->gemms.load(); }
void Context::synchronize() const { impl_->queue.wait_and_throw(); }
std::string Context::device_name() const { return impl_->queue.get_device().get_info<sycl::info::device::name>(); }
Array::Array(const Context& ctx, DType type, std::int64_t count) : ctx_(&ctx), count_(count), type_(type) {
  if (count <= 0 || static_cast<std::uint64_t>(count) > std::numeric_limits<std::size_t>::max() / size_of(type))
    throw std::invalid_argument("invalid device allocation size");
  const auto bytes = static_cast<std::size_t>(count) * size_of(type);
  auto& impl = ctx.impl();
  auto used = impl.allocated.load();
  do {
    if (bytes > impl.cap || used > impl.cap - bytes)
      throw std::runtime_error("Intel device allocation budget exceeded");
  } while (!impl.allocated.compare_exchange_weak(used, used + bytes));
  try {
    data_ = sycl::malloc_device(bytes, impl.queue);
    if (!data_) throw std::bad_alloc();
  } catch (...) {
    impl.allocated.fetch_sub(bytes);
    throw;
  }
  auto peak = impl.peak.load();
  while (peak < used + bytes && !impl.peak.compare_exchange_weak(peak, used + bytes)) {
  }
}
Array::~Array() {
  if (data_) {
    ctx_->impl().queue.wait();
    sycl::free(data_, ctx_->impl().queue);
    ctx_->impl().allocated.fetch_sub(static_cast<std::size_t>(count_) * size_of(type_));
  }
}
Array::Array(Array&& other) noexcept {
  std::swap(ctx_, other.ctx_);
  std::swap(data_, other.data_);
  std::swap(count_, other.count_);
  std::swap(type_, other.type_);
}
Array& Array::operator=(Array&& other) noexcept {
  if (this != &other) {
    Array old(std::move(*this));
    std::swap(ctx_, other.ctx_);
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
    std::swap(type_, other.type_);
  }
  return *this;
}
void Array::upload(const float* host, std::int64_t count, std::int64_t offset) {
  bounds(count_, count, offset);
  if (type_ == DType::kF32)
    ctx_->impl().queue.memcpy(f32() + offset, host, static_cast<std::size_t>(count) * 4).wait_and_throw();
  else if (type_ == DType::kBF16) {
    std::vector<std::uint16_t> b(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) b[static_cast<std::size_t>(i)] = f32_to_bf16(host[i]);
    ctx_->impl().queue.memcpy(bf16() + offset, b.data(), b.size() * 2).wait_and_throw();
  } else
    throw std::invalid_argument("float upload dtype mismatch");
}
void Array::upload(const std::int64_t* host, std::int64_t count, std::int64_t offset) {
  bounds(count_, count, offset);
  ctx_->impl().queue.memcpy(i64() + offset, host, static_cast<std::size_t>(count) * 8).wait_and_throw();
}
void Array::upload(const std::int8_t* host, std::int64_t count, std::int64_t offset) {
  bounds(count_, count, offset);
  ctx_->impl().queue.memcpy(i8() + offset, host, static_cast<std::size_t>(count)).wait_and_throw();
}
std::vector<float> Array::download() const {
  std::vector<float> out(static_cast<std::size_t>(count_));
  if (type_ == DType::kF32)
    ctx_->impl().queue.memcpy(out.data(), data_, out.size() * 4).wait_and_throw();
  else if (type_ == DType::kBF16) {
    std::vector<std::uint16_t> b(out.size());
    ctx_->impl().queue.memcpy(b.data(), data_, b.size() * 2).wait_and_throw();
    for (std::size_t i = 0; i < b.size(); ++i) out[i] = bf16_to_f32(b[i]);
  } else
    throw std::invalid_argument("download dtype mismatch");
  return out;
}
void Array::zero() { ctx_->impl().queue.memset(data_, 0, static_cast<std::size_t>(count_) * size_of(type_)); }
float* Array::f32() const {
  if (type_ != DType::kF32) throw std::invalid_argument("not fp32");
  return static_cast<float*>(data_);
}
std::uint16_t* Array::bf16() const {
  if (type_ != DType::kBF16) throw std::invalid_argument("not bf16");
  return static_cast<std::uint16_t*>(data_);
}
std::int64_t* Array::i64() const {
  if (type_ != DType::kI64) throw std::invalid_argument("not i64");
  return static_cast<std::int64_t*>(data_);
}
std::int8_t* Array::i8() const {
  if (type_ != DType::kI8) throw std::invalid_argument("not i8");
  return static_cast<std::int8_t*>(data_);
}
}  // namespace uocr::intel_native
