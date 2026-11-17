#pragma once

// Intel Level Zero GPU handles and typed USM allocations, without SYCL
// headers in the model interface. No CPU selector is supported.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace uocr::intel_native {

struct DeviceInfo {
  std::string name;
  std::uint64_t total_bytes = 0;
};

// Number of visible devices; 0 when there is no usable Intel SYCL driver.
int device_count();
DeviceInfo device_info(int index);

class Context {
 public:
  explicit Context(int device = 0, std::uint64_t memory_cap = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  void synchronize() const;
  std::string device_name() const;
  std::uint64_t allocated_bytes() const;
  std::uint64_t peak_bytes() const;
  std::uint64_t memory_cap() const;
  std::uint64_t kernel_submissions() const;
  std::uint64_t gemm_submissions() const;
  // One-shot allocation fault injection; -1 disables it. No effect by default.
  void fail_allocation_after_for_testing(std::int64_t attempts) const;
  struct Impl;
  Impl& impl() const { return *impl_; }

 private:
  std::unique_ptr<Impl> impl_;
};

enum class DType : std::uint8_t { kF32, kBF16, kI64, kI8 };

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
  // Host bytes in, for kI8 arrays.
  void upload(const std::int8_t* host, std::int64_t count, std::int64_t offset = 0);
  // Device contents widened to fp32 (synchronises the stream).
  std::vector<float> download() const;
  void zero();

  void* data() const { return data_; }
  float* f32() const;
  std::uint16_t* bf16() const;
  std::int64_t* i64() const;
  std::int8_t* i8() const;
  std::int64_t count() const { return count_; }
  DType type() const { return type_; }
  const Context* context() const { return ctx_; }

 private:
  const Context* ctx_ = nullptr;
  void* data_ = nullptr;
  std::int64_t count_ = 0;
  DType type_ = DType::kF32;
};

}  // namespace uocr::intel_native
