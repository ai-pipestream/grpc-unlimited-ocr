#include "backend/cuda/context.h"
#include "backend/cuda/context_impl.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uocr::cuda {
namespace {
std::size_t element_bytes(DType t) { return t == DType::kBF16 ? 2 : t == DType::kF32 ? 4 : 8; }
}  // namespace

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
  }
}

int device_count() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

DeviceInfo device_info(int index) {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties");
  check(cudaSetDevice(index), "cudaSetDevice");
  std::size_t free_b = 0, total_b = 0;
  check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
  return DeviceInfo{prop.name, prop.major, prop.minor, free_b, total_b};
}

Context::Context(int device) : impl_(std::make_unique<Impl>()) {
  impl_->device = device;
  check(cudaSetDevice(device), "cudaSetDevice");
  check(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking), "cudaStreamCreate");
  check(cublasCreate(&impl_->blas), "cublasCreate");
  check(cublasSetStream(impl_->blas, impl_->stream), "cublasSetStream");
  // Plain fp32 GEMMs stay fp32: no TF32 shortcut on the attention products.
  check(cublasSetMathMode(impl_->blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
  check(cublasLtCreate(&impl_->lt), "cublasLtCreate");
  check(cudaMalloc(&impl_->workspace, impl_->workspace_bytes), "cudaMalloc workspace");
}

Context::~Context() {
  if (!impl_) return;
  cudaStreamSynchronize(impl_->stream);
  cudaFree(impl_->workspace);
  cublasLtDestroy(impl_->lt);
  cublasDestroy(impl_->blas);
  cudaStreamDestroy(impl_->stream);
}

void Context::synchronize() const { check(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize"); }

Array::Array(const Context& ctx, DType type, std::int64_t count) : ctx_(&ctx), count_(count), type_(type) {
  const std::size_t bytes = static_cast<std::size_t>(count) * element_bytes(type);
  if (bytes > 0) check(cudaMallocAsync(&data_, bytes, ctx.impl().stream), "cudaMallocAsync");
}

Array::~Array() {
  if (data_ != nullptr) cudaFreeAsync(data_, ctx_->impl().stream);
}

Array::Array(Array&& o) noexcept
    : ctx_(o.ctx_), data_(std::exchange(o.data_, nullptr)), count_(std::exchange(o.count_, 0)), type_(o.type_) {}

Array& Array::operator=(Array&& o) noexcept {
  if (this != &o) {
    if (data_ != nullptr) cudaFreeAsync(data_, ctx_->impl().stream);
    ctx_ = o.ctx_;
    data_ = std::exchange(o.data_, nullptr);
    count_ = std::exchange(o.count_, 0);
    type_ = o.type_;
  }
  return *this;
}

float* Array::f32() const {
  if (type_ != DType::kF32) throw std::logic_error("Array: not fp32");
  return static_cast<float*>(data_);
}

std::int64_t* Array::i64() const {
  if (type_ != DType::kI64) throw std::logic_error("Array: not int64");
  return static_cast<std::int64_t*>(data_);
}

void Array::upload(const std::int64_t* host, std::int64_t count, std::int64_t offset) {
  if (offset < 0 || offset + count > count_) throw std::out_of_range("Array::upload past the end");
  auto stream = ctx_->impl().stream;
  check(cudaMemcpyAsync(i64() + offset, host, static_cast<std::size_t>(count) * 8, cudaMemcpyHostToDevice, stream),
        "upload i64");
  check(cudaStreamSynchronize(stream), "upload sync");
}

std::uint16_t* Array::bf16() const {
  if (type_ != DType::kBF16) throw std::logic_error("Array: not bf16");
  return static_cast<std::uint16_t*>(data_);
}

void Array::upload(const float* host, std::int64_t count, std::int64_t offset) {
  if (offset < 0 || offset + count > count_) throw std::out_of_range("Array::upload past the end");
  auto stream = ctx_->impl().stream;
  if (type_ == DType::kF32) {
    check(cudaMemcpyAsync(f32() + offset, host, static_cast<std::size_t>(count) * 4, cudaMemcpyHostToDevice, stream),
          "upload f32");
    check(cudaStreamSynchronize(stream), "upload sync");
    return;
  }
  std::vector<__nv_bfloat16> staged(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i) staged[static_cast<std::size_t>(i)] = __float2bfloat16_rn(host[i]);
  check(cudaMemcpyAsync(bf16() + offset, staged.data(), static_cast<std::size_t>(count) * 2, cudaMemcpyHostToDevice,
                        stream),
        "upload bf16");
  check(cudaStreamSynchronize(stream), "upload sync");
}

std::vector<float> Array::download() const {
  if (type_ == DType::kI64) throw std::logic_error("Array::download: int64 arrays are not widened to fp32");
  auto stream = ctx_->impl().stream;
  std::vector<float> out(static_cast<std::size_t>(count_));
  if (type_ == DType::kF32) {
    check(cudaMemcpyAsync(out.data(), data_, out.size() * 4, cudaMemcpyDeviceToHost, stream), "download f32");
    check(cudaStreamSynchronize(stream), "download sync");
    return out;
  }
  std::vector<__nv_bfloat16> staged(out.size());
  check(cudaMemcpyAsync(staged.data(), data_, out.size() * 2, cudaMemcpyDeviceToHost, stream), "download bf16");
  check(cudaStreamSynchronize(stream), "download sync");
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(staged[i]);
  return out;
}

void Array::zero() {
  check(cudaMemsetAsync(data_, 0, static_cast<std::size_t>(count_) * element_bytes(type_),
                        ctx_->impl().stream),
        "zero");
}

}  // namespace uocr::cuda
