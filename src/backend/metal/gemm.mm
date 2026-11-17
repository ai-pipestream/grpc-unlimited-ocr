#include "backend/metal/gemm.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace uocr::metal {
struct Gemm::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  std::size_t cap;
  std::mutex mutex;
};

unsigned Gemm::device_count() {
  @autoreleasepool { return static_cast<unsigned>([MTLCopyAllDevices() count]); }
}

Gemm::Gemm(unsigned index, std::size_t cap) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (index >= devices.count) throw std::runtime_error("Metal device index unavailable");
    impl_->device = devices[index];
    if (!impl_->device.hasUnifiedMemory || !MPSSupportsMTLDevice(impl_->device))
      throw std::runtime_error("MPS GEMM requires a unified-memory Apple GPU");
    impl_->queue = [impl_->device newCommandQueueWithMaxCommandBufferCount:1];
    if (!impl_->queue) throw std::runtime_error("Metal command queue allocation failed");
    impl_->cap = cap;
  }
}
Gemm::~Gemm() = default;
std::string Gemm::device_name() const { return [impl_->device.name UTF8String]; }
std::string Gemm::device_id() const { return std::to_string(impl_->device.registryID); }

Tensor Gemm::linear(const Tensor& x, const Tensor& w) {
  if (x.rank() == 0 || w.rank() != 2 || x.shape().back() <= 0 ||
      w.dim(0) <= 0 || w.dim(1) != x.shape().back() || x.numel() <= 0)
    throw std::invalid_argument("Metal linear requires nonempty [..., K] and [N, K]");
  const auto m = static_cast<std::size_t>(x.rows());
  const auto n = static_cast<std::size_t>(w.dim(0));
  const auto k = static_cast<std::size_t>(w.dim(1));
  // Division guards precede products, including the output allocation.
  const auto elements = impl_->cap / sizeof(float);
  if (m > elements / k || n > elements / k || m > elements / n)
    throw std::length_error("Metal GEMM exceeds buffer budget");
  const auto a_bytes = m * k * sizeof(float), b_bytes = n * k * sizeof(float);
  const auto c_bytes = m * n * sizeof(float);
  if (a_bytes > impl_->cap - b_bytes || c_bytes > impl_->cap - a_bytes - b_bytes)
    throw std::length_error("Metal GEMM exceeds aggregate buffer budget");
  std::lock_guard lock(impl_->mutex);
  @autoreleasepool {
    auto device = impl_->device;
    id<MTLBuffer> a = [device newBufferWithBytes:x.data() length:a_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithBytes:w.data() length:b_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
    if (!a || !b || !c) throw std::runtime_error("Metal GEMM buffer allocation failed");
    auto matrix = [](id<MTLBuffer> buffer, std::size_t rows, std::size_t cols) {
      auto descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
                           rowBytes:cols * sizeof(float) dataType:MPSDataTypeFloat32];
      return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    };
    auto multiply = [[MPSMatrixMultiplication alloc] initWithDevice:device
      transposeLeft:false transposeRight:true resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> command = [impl_->queue commandBuffer];
    if (!command || !multiply) throw std::runtime_error("Metal GEMM command allocation failed");
    [multiply encodeToCommandBuffer:command leftMatrix:matrix(a, m, k)
      rightMatrix:matrix(b, n, k) resultMatrix:matrix(c, m, n)];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error("Metal GEMM command failed");
    auto shape = x.shape();
    shape.back() = static_cast<std::int64_t>(n);
    Tensor out(std::move(shape));
    std::memcpy(out.data(), c.contents, c_bytes);
    return out;
  }
}
}  // namespace uocr::metal
