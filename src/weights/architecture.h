#pragma once

// The Unlimited-OCR 2 architecture as code: its dimensions, and the exact
// tensor table a checkpoint must hold for the engine to run it. Startup
// checks a checkpoint against the table so a wrong or truncated download
// fails with the tensor's name instead of producing garbage text.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "tensor/dtype.h"

namespace uocr {

class ShardedWeights;

struct DecoderDims {
  std::int64_t vocab = 129280;
  std::int64_t hidden = 1280;
  std::int64_t layers = 12;
  std::int64_t heads = 10;
  std::int64_t head_dim = 128;
  std::int64_t dense_ffn = 6848;     // layer 0
  std::int64_t first_moe_layer = 1;  // first_k_dense_replace
  std::int64_t routed_experts = 64;
  std::int64_t experts_per_token = 6;
  std::int64_t shared_experts = 2;
  std::int64_t expert_ffn = 896;
  std::int64_t index_dim = 128;  // sparse_index_dim
  double rms_norm_eps = 1e-6;
};

struct SamDims {
  std::int64_t image = 1024;
  std::int64_t patch = 16;
  std::int64_t width = 768;
  std::int64_t layers = 12;
  std::int64_t heads = 12;
  std::int64_t mlp = 3072;
  std::int64_t window = 14;
  std::array<std::int64_t, 4> global_layers = {2, 5, 8, 11};
  std::int64_t neck = 256;
  std::int64_t net2 = 512;
  std::int64_t net3 = 1024;
};

struct ClipDims {
  std::int64_t width = 1024;
  std::int64_t layers = 24;
  std::int64_t heads = 16;
  std::int64_t mlp = 4096;
  std::int64_t patch = 14;
  std::int64_t positions = 257;  // 16 x 16 patches + class token
};

struct ModelDims {
  DecoderDims decoder;
  SamDims sam;
  ClipDims clip;
  std::int64_t projector_in = 2048;  // CLIP 1024 + SAM 1024, concatenated
};

struct TensorSpec {
  std::string name;
  DType dtype;
  std::vector<std::int64_t> shape;
};

// Every tensor of the published Unlimited-OCR 2 checkpoint, in a fixed
// order.
std::vector<TensorSpec> unlimited_ocr2_tensors(const ModelDims& dims = {});

// Problems found when checking `weights` against `specs`: each missing
// tensor, each dtype or shape mismatch, and each tensor the table does not
// expect. Empty means the checkpoint is exactly the architecture.
std::vector<std::string> check_architecture(const std::vector<TensorSpec>& specs, const ShardedWeights& weights);

}  // namespace uocr
