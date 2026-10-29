#pragma once

// CUDA-side view of a Context, for .cu files only.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "backend/cuda/context.h"

namespace uocr::cuda {

struct Context::Impl {
  int device = 0;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cublasLtHandle_t lt = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 32u << 20;
};

void check(cudaError_t status, const char* what);
void check(cublasStatus_t status, const char* what);

}  // namespace uocr::cuda
