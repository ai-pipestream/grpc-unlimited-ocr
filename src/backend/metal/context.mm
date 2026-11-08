#include "backend/metal/context_impl.h"
#include "tensor/dtype.h"
#include "metal_source.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace uocr::metal {
namespace {
struct EndEncoding {
  id<MTLComputeCommandEncoder> encoder;
  ~EndEncoding() { [encoder endEncoding]; }
};
}
int device_count() { @autoreleasepool { return static_cast<int>(MTLCopyAllDevices().count); } }
DeviceInfo device_info(int index) {
  @autoreleasepool {
    auto devices = MTLCopyAllDevices();
    if (index < 0 || static_cast<NSUInteger>(index) >= devices.count) throw std::invalid_argument("Metal device index");
    id<MTLDevice> d = devices[static_cast<NSUInteger>(index)];
    return {d.name.UTF8String, d.recommendedMaxWorkingSetSize};
  }
}
Context::Context(int index, std::size_t cap) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    auto devices = MTLCopyAllDevices();
    if (index < 0 || static_cast<NSUInteger>(index) >= devices.count) throw std::invalid_argument("Metal device index");
    impl_->device = devices[static_cast<NSUInteger>(index)];
    if (!impl_->device.hasUnifiedMemory) throw std::runtime_error("Metal requires Apple unified memory");
    impl_->cap = std::min(cap, static_cast<std::size_t>(impl_->device.recommendedMaxWorkingSetSize));
    impl_->queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:2];
    impl_->dummy = [impl_->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* error = nil;
    impl_->library = [impl_->device newLibraryWithSource:@(kMetalSource) options:options error:&error];
    if (!impl_->queue || !impl_->dummy || !impl_->library)
      throw std::runtime_error(error ? error.localizedDescription.UTF8String : "Metal initialization failed");
  }
}
Context::~Context() { try { synchronize(); } catch (...) {} }
void Context::synchronize() const { @autoreleasepool { impl_->finish(); } }
std::size_t Context::allocated_bytes() const { return impl_->allocated; }
std::size_t Context::peak_bytes() const { return impl_->peak; }
std::size_t Context::device_allocated_bytes() const { return impl_->device.currentAllocatedSize; }
std::uint64_t Context::kernel_dispatches() const { return impl_->kernels; }
std::uint64_t Context::gemm_dispatches() const { return impl_->gemms; }
Binding Context::Impl::binding(const void* ptr) const {
  if (!ptr) return {dummy, 0};
  const auto address = reinterpret_cast<std::uintptr_t>(ptr);
  auto it = buffers.upper_bound(address);
  if (it == buffers.begin()) throw std::invalid_argument("pointer is not a Metal allocation");
  --it;
  const auto offset = address - it->first;
  if (offset >= it->second.length) throw std::out_of_range("Metal buffer slice");
  return {it->second, offset};
}
id<MTLCommandBuffer> Context::Impl::commands() {
  if (failure) std::rethrow_exception(failure);
  if (!command) command = [queue commandBuffer];
  if (!command) throw std::runtime_error("Metal command allocation failed");
  return command;
}
void Context::Impl::finish() {
  if (!command) {
    allocated -= retired_bytes;
    retired_bytes = 0;
    retired.clear();
    if (failure) std::rethrow_exception(failure);
    return;
  }
  auto submitted = command;
  command = nil;
  pending = 0;
  [submitted commit];
  [submitted waitUntilCompleted];
  allocated -= retired_bytes;
  retired_bytes = 0;
  retired.clear();
  if (!failure && submitted.status != MTLCommandBufferStatusCompleted) {
    failure = std::make_exception_ptr(std::runtime_error(submitted.error ? submitted.error.localizedDescription.UTF8String : "Metal execution failed"));
  }
  if (failure) std::rethrow_exception(failure);
}
void dispatch(const Context& ctx, const char* name, std::size_t threads,
              std::initializer_list<const void*> buffers, const Params& params) {
  if (!threads) return;
  @autoreleasepool {
    auto& p = ctx.impl();
    // Reject invalid pointers before opening an encoder. ARC releasing an
    // encoder does not close it, so a bounds error must not leave one active.
    if (buffers.size() > 8) throw std::invalid_argument("Metal buffer slots exceed scalar ABI");
    std::array<Binding, 8> bindings{};
    std::size_t count = 0;
    for (auto ptr : buffers) bindings[count++] = p.binding(ptr);
    auto& pipeline = p.pipelines[name];
    if (!pipeline) {
      auto function = [p.library newFunctionWithName:@(name)];
      NSError* error = nil;
      pipeline = [p.device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) throw std::runtime_error(error ? error.localizedDescription.UTF8String : name);
    }
    if (pipeline.threadExecutionWidth != 32) throw std::runtime_error("Metal SIMD width must be 32");
    {
      auto encoder = [p.commands() computeCommandEncoder];
      if (!encoder) throw std::runtime_error("Metal compute encoder allocation failed");
      const EndEncoding end{encoder};
      [encoder setComputePipelineState:pipeline];
      for (std::size_t slot = 0; slot < count; ++slot) {
        const auto& b = bindings[slot];
        [encoder setBuffer:b.buffer offset:b.offset atIndex:slot];
      }
      [encoder setBytes:&params length:sizeof(params) atIndex:8];
      const auto group = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    }
    ++p.kernels;
    if (++p.pending >= 64) p.finish();
  }
}
namespace {
std::size_t element_size(DType t) {
  switch (t) {
    case DType::kF32: return 4;
    case DType::kBF16: return 2;
    case DType::kI64: return 8;
  }
  throw std::invalid_argument("unsupported Metal array dtype");
}
}
Array::Array(const Context& ctx, DType type, std::int64_t count) : ctx_(&ctx), count_(count), type_(type) {
  if (count <= 0) throw std::invalid_argument("Metal array must be nonempty");
  auto& p = ctx.impl();
  const auto size = element_size(type);
  if (static_cast<std::uint64_t>(count) > p.cap / size) throw std::length_error("Metal allocation exceeds cap");
  const auto bytes = static_cast<std::size_t>(count) * size;
  if (bytes > p.cap - p.allocated) p.finish();
  if (bytes > p.cap - p.allocated) throw std::length_error("Metal aggregate allocation exceeds cap");
  @autoreleasepool {
    id<MTLBuffer> buffer = [p.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) throw std::bad_alloc();
    data_ = buffer.contents;
    p.buffers.emplace(reinterpret_cast<std::uintptr_t>(data_), buffer);
  }
  p.allocated += bytes;
  p.peak = std::max(p.peak, p.allocated);
}
void Array::release() noexcept {
  if (!data_) return;
  auto& p = ctx_->impl();
  const auto it = p.buffers.find(reinterpret_cast<std::uintptr_t>(data_));
  const auto bytes = static_cast<std::size_t>(count_) * element_size(type_);
  if (p.command) {
    // Retain storage and its budget until all encoded users finish. No
    // per-op host wait and no reuse of memory still referenced by the GPU.
    // Allocation failure here must not escape a destructor.
    try { p.retired.push_back(it->second); p.retired_bytes += bytes; }
    catch (...) {
      try { p.finish(); } catch (...) {}
      p.allocated -= bytes;
    }
  } else {
    p.allocated -= bytes;
  }
  p.buffers.erase(it);
  data_ = nullptr;
}
Array::~Array() { release(); }
Array::Array(Array&& o) noexcept { *this = std::move(o); }
Array& Array::operator=(Array&& o) noexcept {
  if (this != &o) { release(); ctx_ = o.ctx_; data_ = std::exchange(o.data_, nullptr); count_ = o.count_; type_ = o.type_; }
  return *this;
}
void Array::bounds(std::int64_t n, std::int64_t offset) const {
  if (n < 0 || offset < 0 || offset > count_ || n > count_ - offset) throw std::out_of_range("Metal upload bounds");
}
void Array::upload(const float* host, std::int64_t n, std::int64_t offset) {
  bounds(n, offset); ctx_->synchronize();
  if (type_ == DType::kF32) std::memcpy(f32() + offset, host, static_cast<std::size_t>(n) * 4);
  else if (type_ == DType::kBF16) for (std::int64_t i = 0; i < n; ++i) bf16()[offset + i] = f32_to_bf16(host[i]);
  else throw std::invalid_argument("float upload into integer array");
}
void Array::upload(const std::int64_t* host, std::int64_t n, std::int64_t offset) {
  bounds(n, offset); ctx_->synchronize();
  std::memcpy(i64() + offset, host, static_cast<std::size_t>(n) * 8);
}
std::vector<float> Array::download() const {
  ctx_->synchronize(); std::vector<float> out(static_cast<std::size_t>(count_));
  if (type_ == DType::kF32) std::memcpy(out.data(), data_, out.size() * 4);
  else if (type_ == DType::kBF16) for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_f32(bf16()[i]);
  else throw std::invalid_argument("integer download as float");
  return out;
}
void Array::zero() { ctx_->synchronize(); std::memset(data_, 0, static_cast<std::size_t>(count_) * element_size(type_)); }
float* Array::f32() const { if(type_ != DType::kF32) throw std::logic_error("not fp32"); return static_cast<float*>(data_); }
std::uint16_t* Array::bf16() const { if(type_ != DType::kBF16) throw std::logic_error("not bf16"); return static_cast<std::uint16_t*>(data_); }
std::int64_t* Array::i64() const { if(type_ != DType::kI64) throw std::logic_error("not int64"); return static_cast<std::int64_t*>(data_); }
}  // namespace uocr::metal
