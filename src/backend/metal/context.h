#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace uocr::metal {
struct DeviceInfo {
  std::string name;
  std::uint64_t total_bytes = 0;
};
int device_count();
DeviceInfo device_info(int index);
// A serialized native queue. Raw pointers below address shared MTLBuffers;
// only transfer/grouping metadata is read by the CPU, never model arithmetic.
class Context {
 public:
  explicit Context(int device = 0, std::size_t byte_cap = 12ULL << 30);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  void synchronize() const;
  std::size_t allocated_bytes() const;
  std::size_t peak_bytes() const;
  std::size_t device_allocated_bytes() const;
  std::uint64_t kernel_dispatches() const;
  std::uint64_t gemm_dispatches() const;
  struct Impl;
  Impl& impl() const { return *impl_; }
 private:
  std::unique_ptr<Impl> impl_;
};
enum class DType : std::uint8_t { kF32, kBF16, kI64 };
class Array {
 public:
  Array() = default;
  Array(const Context& ctx, DType type, std::int64_t count);
  ~Array();
  Array(Array&& other) noexcept;
  Array& operator=(Array&& other) noexcept;
  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;
  void upload(const float* host, std::int64_t count, std::int64_t offset = 0);
  void upload(const std::int64_t* host, std::int64_t count, std::int64_t offset = 0);
  std::vector<float> download() const;
  void zero();
  void* data() const { return data_; }
  float* f32() const;
  std::uint16_t* bf16() const;
  std::int64_t* i64() const;
  std::int64_t count() const { return count_; }
  DType type() const { return type_; }
 private:
  void release() noexcept;
  void bounds(std::int64_t count, std::int64_t offset) const;
  const Context* ctx_ = nullptr;
  void* data_ = nullptr;
  std::int64_t count_ = 0;
  DType type_ = DType::kF32;
};
}  // namespace uocr::metal
