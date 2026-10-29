#pragma once

// The CUDA backend's device handles and memory, behind a header with no
// CUDA includes: host code built with the newest C++ compiler links
// against kernels built by nvcc without sharing any CUDA headers.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace uocr::cuda {

struct DeviceInfo {
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::uint64_t free_bytes = 0;
  std::uint64_t total_bytes = 0;
};

// Number of visible devices; 0 when there is no usable CUDA driver.
int device_count();
DeviceInfo device_info(int index);

// One device, one stream, the cuBLAS handles and a GEMM workspace. Every
// op runs on this stream in order; host reads synchronise explicitly.
class Graph;

class Context {
 public:
  explicit Context(int device = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  void synchronize() const;

  // Records everything enqueued between the two calls as a graph instead
  // of running it.
  void begin_capture() const;
  std::unique_ptr<Graph> end_capture() const;

  struct Impl;
  Impl& impl() const { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

// A captured, instantiated sequence of device work, replayed on the
// context's stream.
class Graph {
 public:
  ~Graph();
  void launch() const;

  struct Impl;
  explicit Graph(std::unique_ptr<Impl> impl);

 private:
  std::unique_ptr<Impl> impl_;
};

// Page-locked host memory, so copies to and from the device are enqueued
// without blocking the host.
class PinnedBuffer {
 public:
  explicit PinnedBuffer(std::size_t bytes);
  ~PinnedBuffer();
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
};

// Stream-ordered copies; neither waits.
void copy_to_device(const Context& ctx, void* device_dst, const PinnedBuffer& src, std::size_t bytes);
void copy_to_host(const Context& ctx, PinnedBuffer& dst, const void* device_src, std::size_t bytes);

enum class DType : std::uint8_t { kF32, kBF16, kI64 };

// Device memory of `count` elements of one type. Contents are undefined
// until written.
class Array {
 public:
  Array() = default;
  Array(const Context& ctx, DType type, std::int64_t count);
  ~Array();
  Array(Array&& other) noexcept;
  Array& operator=(Array&& other) noexcept;
  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;

  // Host fp32 in; stored as `type()` (bf16 rounds to nearest even).
  void upload(const float* host, std::int64_t count, std::int64_t offset = 0);
  // Host indices in, for kI64 arrays.
  void upload(const std::int64_t* host, std::int64_t count, std::int64_t offset = 0);
  // Device contents widened to fp32 (synchronises the stream).
  std::vector<float> download() const;
  void zero();

  void* data() const { return data_; }
  float* f32() const;
  std::uint16_t* bf16() const;
  std::int64_t* i64() const;
  std::int64_t count() const { return count_; }
  DType type() const { return type_; }
  const Context* context() const { return ctx_; }

 private:
  const Context* ctx_ = nullptr;
  void* data_ = nullptr;
  std::int64_t count_ = 0;
  DType type_ = DType::kF32;
};

}  // namespace uocr::cuda
