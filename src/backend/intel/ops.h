#pragma once

// Initial native Intel kernel layer. No inference runtime. This is not yet
// wired into the model/service and must not advertise an Intel model backend.
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace uocr::intel {
class Context {
 public:
  explicit Context(unsigned device = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  std::string device_name() const;
  std::string driver_version() const;
  void synchronize() const;
  struct Impl;
  Impl& impl() const { return *impl_; }
 private:
  std::unique_ptr<Impl> impl_;
};
// Device allocations retain their context; caller must keep it alive.
class Array {
 public:
  Array(const Context&, std::size_t count);
  ~Array();
  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;
  void upload(std::span<const float>);
  std::vector<float> download() const;
  float* data() const { return data_; }
  std::size_t count() const { return count_; }
  const Context& context() const { return context_; }
 private:
  const Context& context_;
  std::size_t count_;
  float* data_;
};
enum class Activation { silu, gelu_erf, quick_gelu };
void activate(Array&, Activation);
void rms_norm(Array&, const Array& weight, std::size_t width, float epsilon);
void softmax_rows(Array&, std::size_t width);
// Row-major x[M,K] times w[N,K] transpose plus optional bias[N].
void linear(const Array& x, const Array& weight, const Array* bias,
            Array& output, std::size_t m, std::size_t n, std::size_t k);
}  // namespace uocr::intel
