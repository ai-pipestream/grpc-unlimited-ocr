#pragma once

// The decoder on the CUDA backend. Tensor math runs on the device; the
// sparse-attention plan (block selection, key masks) and the expert
// routing are decided on the host with the same code the reference
// backend uses, from small score tensors copied back each layer.

#include <optional>
#include <vector>

#include "backend/cuda/context.h"
#include "model/attention_plan.h"
#include "model/block_router.h"
#include "model/cuda/device_params.h"
#include "model/decoder_model.h"
#include "model/page_rope.h"
#include "weights/architecture.h"

namespace uocr::cuda {

class CudaDecoder final : public DecoderModel {
 public:
  CudaDecoder(const Context& ctx, const WeightSource& weights, const DecoderDims& dims = {},
              RouterParams router = default_router_params());

  Tensor embed(const std::vector<std::int64_t>& ids) const override;
  std::vector<float> prefill(const Tensor& embeds, const PromptLayout& layout) override;
  std::vector<float> step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                          const BlockPages& prompt_blocks) override;
  std::int64_t block_size() const override { return router_.block_size; }

  static RouterParams default_router_params();

 private:
  struct Expert {
    DeviceMatrix gate, up, down;
  };
  struct Layer {
    Array input_norm, post_norm, router_q_norm, router_k_norm, head_cls, moe_gate_f32;
    DeviceMatrix q, k, v, o, router_q, router_k;
    std::optional<Expert> dense;
    std::vector<Expert> experts;
    std::optional<Expert> shared;
  };
  struct Cache {
    Array k, v, router_k;  // [capacity, width] each
  };

  // Runs every layer over x [rows, hidden] for queries at positions
  // first_pos.., returning the logits of the last row.
  std::vector<float> run(Array& x, std::int64_t rows, std::int64_t first_pos, const std::vector<RopeCoord>& coords,
                         const std::vector<std::vector<std::uint8_t>>& candidates);
  void attention(const Layer& layer, std::size_t index, const Array& h, std::int64_t rows, std::int64_t first_pos,
                 const Array& tok, const Array& page, const std::vector<std::vector<std::uint8_t>>& candidates,
                 Array& out);
  void mlp(const Expert& e, const Array& x, std::int64_t rows, Array& out) const;
  void moe(const Layer& layer, const Array& h, std::int64_t rows, Array& out) const;
  void reserve(std::int64_t keys);

  const Context& ctx_;
  DecoderDims dims_;
  RouterParams router_;
  PageRope rope_;
  Array embed_, final_norm_, inv_tok_, inv_page_;
  DeviceMatrix lm_head_;
  std::vector<Layer> layers_;
  std::vector<Cache> cache_;
  std::int64_t capacity_ = 0;
  std::int64_t length_ = 0;
};

}  // namespace uocr::cuda
