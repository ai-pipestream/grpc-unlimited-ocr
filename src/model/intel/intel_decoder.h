#pragma once

// Native Intel decoder. Prefill and decode use device MoE selection, block
// routing, projections and attention. Only prompt layout and logit policy
// run on the host. No CPU compute backend is selected by this class.

#include <memory>
#include <optional>
#include <vector>

#include "backend/intel/native/context.h"
#include "model/attention_plan.h"
#include "model/block_router.h"
#include "model/decoder_model.h"
#include "model/intel/device_params.h"
#include "model/intel/device_weight.h"
#include "model/page_rope.h"
#include "weights/architecture.h"

namespace uocr::intel_native {

class IntelDecoder final : public DecoderModel {
 public:
  IntelDecoder(const Context& ctx, const WeightSource& weights, const DecoderDims& dims = {},
               RouterParams router = default_router_params(), WeightFormat format = WeightFormat::kBf16);

  Tensor embed(const std::vector<std::int64_t>& ids) const override;
  std::vector<float> prefill(const Tensor& embeds, const PromptLayout& layout) override;
  std::vector<float> step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                          const BlockPages& prompt_blocks) override;
  std::int64_t block_size() const override { return router_.block_size; }

  static RouterParams default_router_params();

 private:
  struct Expert {
    DeviceWeight gate, up, down;
  };
  // The routed experts of one layer, stacked by expert so a kernel can pick
  // them by id: gate/up [experts * ffn, hidden], down [experts * hidden, ffn].
  struct ExpertStack {
    DeviceWeight gate, up, down;
    std::int64_t ffn = 0;
  };
  struct Layer {
    Array input_norm, post_norm, router_q_norm, router_k_norm, head_cls, moe_gate_f32;
    DeviceWeight q, k, v, o, router_q, router_k;
    std::optional<Expert> dense;
    ExpertStack experts;
    std::optional<Expert> shared;
  };
  struct Cache {
    Array k, v, router_k;  // [capacity, width] each
    Array summaries;       // router block summaries [capacity blocks, index dim]
  };
  // Runs every layer over x [rows, hidden] for queries at positions
  // first_pos.., returning the logits of the last row.
  // Each query's page for routing: per row on the device (prefill) or one
  // value for the single decode row.
  struct QueryPages {
    const std::int64_t* rows = nullptr;
    std::int64_t one = -1;
  };
  std::vector<float> run(Array& x, std::int64_t rows, std::int64_t first_pos, const std::vector<RopeCoord>& coords,
                         QueryPages pages);
  void attention(const Layer& layer, std::size_t index, const Array& h, std::int64_t rows, std::int64_t first_pos,
                 const Array& tok, const Array& page, QueryPages pages, Array& out);
  void mlp(const Expert& e, const Array& x, std::int64_t rows, Array& out) const;
  void mlp(const ExpertStack& stack, std::int64_t expert, const Array& x, std::int64_t rows, Array& out) const;
  void moe(const Layer& layer, const Array& h, std::int64_t rows, Array& out) const;
  void reserve(std::int64_t keys);
  static ExpertStack stack_experts(const Context& ctx, const WeightSource& w, const std::string& prefix,
                                   const DecoderDims& dims, WeightFormat format);

  const Context& ctx_;
  DecoderDims dims_;
  RouterParams router_;
  PageRope rope_;
  Array embed_, final_norm_, inv_tok_, inv_page_;
  DeviceWeight lm_head_;
  std::vector<Layer> layers_;
  std::vector<Cache> cache_;
  Array block_min_, block_max_;  // image-page range per prompt block
  std::int64_t prompt_blocks_ = 0;
  std::int64_t capacity_ = 0;
  std::int64_t capacity_blocks_ = 0;
  std::int64_t length_ = 0;
  RopeCoord one_coord_;  // the coordinate of a single-row call
};

}  // namespace uocr::intel_native
