#pragma once

// The vision encoder on the CUDA backend: the same SAM, CLIP and projector
// as model/vision_encoder.h, running on the device with bf16 weights and
// fp32 activations.

#include <map>
#include <tuple>
#include <vector>

#include "backend/cuda/context.h"
#include "model/cuda/device_params.h"
#include "model/vision_encoder.h"
#include "weights/architecture.h"

namespace uocr::cuda {

class CudaVisionEncoder final : public VisionModel {
 public:
  CudaVisionEncoder(const Context& ctx, const WeightSource& weights, const ModelDims& dims = {});

  Tensor encode_view(const Tensor& image) const override;

 private:
  struct SamBlock {
    Array norm1_w, norm1_b, qkv_b, proj_b, norm2_w, norm2_b, lin1_b, lin2_b;
    DeviceMatrix qkv, proj, lin1, lin2;
    Tensor rel_h, rel_w;  // learned tables; resized per span on the host
    bool global = false;
  };
  struct ClipLayer {
    Array ln1_w, ln1_b, qkv_b, out_b, ln2_w, ln2_b, fc1_b, fc2_b;
    DeviceMatrix qkv, out, fc1, fc2;
  };

  // SAM over one image: cells [g4 * g4, 1024] in row order.
  Array sam_forward(const Array& image, std::int64_t size, std::int64_t& grid) const;
  void sam_block(const SamBlock& block, Array& x, std::int64_t g) const;
  void sam_attention(const SamBlock& block, const Array& x, std::int64_t windows, std::int64_t h, std::int64_t w,
                     Array& out) const;
  Array clip_forward(const Array& cells, std::int64_t grid) const;
  // Per-grid tables computed once: SAM positions, CLIP positions, rel-pos tables.
  const Array& sam_positions(std::int64_t grid) const;
  const Array& clip_positions(std::int64_t grid) const;
  const Array& rel_table(const SamBlock& block, bool height, std::int64_t span) const;

  const Context& ctx_;
  ModelDims dims_;
  DeviceMatrix patch_, neck0_, neck2_, net2_, net3_, projector_;
  Array patch_b_, neck1_w_, neck1_b_, neck3_w_, neck3_b_, projector_b_, class_embedding_;
  Array pre_ln_w_, pre_ln_b_;
  Tensor sam_pos_host_, clip_pos_host_;
  std::vector<SamBlock> sam_blocks_;
  std::vector<ClipLayer> clip_layers_;
  mutable std::map<std::int64_t, Array> sam_pos_, clip_pos_;
  mutable std::map<std::tuple<const SamBlock*, bool, std::int64_t>, Array> rel_tables_;
};

}  // namespace uocr::cuda
