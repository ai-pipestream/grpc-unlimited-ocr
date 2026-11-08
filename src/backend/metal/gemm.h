#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include "tensor/tensor.h"

namespace uocr::metal {
// Dense fp32 GEMM only. This is not an inference runtime or a complete
// model backend. Inputs/outputs are host tensors; transfers are explicit.
class Gemm {
 public:
  explicit Gemm(unsigned device_index = 0, std::size_t byte_cap = 256 * 1024 * 1024);
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;
  static unsigned device_count();
  std::string device_name() const;
  std::string device_id() const;
  Tensor linear(const Tensor& x, const Tensor& weight);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace uocr::metal
