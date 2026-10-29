#pragma once

// Model parameters on the device: matrices in bf16 (the checkpoint's own
// precision, so the upload is exact), vectors and tables in fp32.

#include <string>

#include "backend/cuda/context.h"
#include "model/weight_source.h"
#include "tensor/tensor.h"

namespace uocr::cuda {

inline Array upload_tensor(const Context& ctx, const Tensor& t, DType type) {
  Array a(ctx, type, t.numel());
  a.upload(t.data(), t.numel());
  return a;
}

// A weight matrix [out, in] (any trailing shape flattened into `in`).
struct DeviceMatrix {
  Array data;
  std::int64_t rows = 0;
  std::int64_t cols = 0;

  DeviceMatrix() = default;
  DeviceMatrix(const Context& ctx, const Tensor& t)
      : data(upload_tensor(ctx, t, DType::kBF16)), rows(t.dim(0)), cols(t.numel() / t.dim(0)) {}
};

inline DeviceMatrix matrix(const Context& ctx, const WeightSource& w, const std::string& name) {
  return DeviceMatrix(ctx, w.get(name));
}

inline Array vector(const Context& ctx, const WeightSource& w, const std::string& name) {
  return upload_tensor(ctx, w.get(name), DType::kF32);
}

}  // namespace uocr::cuda
