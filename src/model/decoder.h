#pragma once

// The language decoder: 12 layers of pre-norm attention with two-axis rope
// and block-sparse key selection, each followed by a gated MLP (layer 0)
// or a mixture of experts. Runs one prompt (prefill) and then one token at
// a time (decode) against a growing cache.

#include <cstdint>
#include <vector>

#include "model/block_router.h"
#include "model/feed_forward.h"
#include "model/page_layout.h"
#include "model/page_rope.h"
#include "model/weight_source.h"
#include "tensor/tensor.h"
#include "weights/architecture.h"

namespace uocr {

// Keys, values and router keys of every position seen so far, per layer.
struct DecoderCache {
  struct Layer {
    std::vector<float> k, v, router_k;
  };
  std::vector<Layer> layers;
  std::int64_t length = 0;
};

class Decoder {
 public:
  Decoder(const WeightSource& weights, const DecoderDims& dims = {}, RouterParams router = default_router());

  static RouterParams default_router();

  // Token embeddings [ids, hidden].
  Tensor embed(const std::vector<std::int64_t>& ids) const;

  // Runs the prompt embeddings [S, hidden] through every layer, filling
  // `cache`; returns the logits of the last position.
  std::vector<float> prefill(const Tensor& embeds, const PromptLayout& layout, DecoderCache& cache) const;

  // Feeds one token at rope coordinate `coord`, querying on response page
  // `query_page` (-1 = unconstrained); returns the next-token logits.
  std::vector<float> step(std::int64_t token, RopeCoord coord, std::int64_t query_page, const BlockPages& prompt_blocks,
                          DecoderCache& cache) const;

  const DecoderDims& dims() const { return dims_; }

 private:
  struct Router {
    Tensor q, k, q_norm, k_norm, head_cls;
  };
  struct Layer {
    Tensor input_norm, post_norm, q, k, v, o;
    Router router;
    FeedForward ffn;
  };

  // One layer over queries x [L, hidden] at absolute positions starting at
  // cache.length, given each query's candidate blocks ([L, nblk] after the
  // new keys are appended; empty = all).
  Tensor attention(const Layer& layer, const Tensor& x, const std::vector<RopeCoord>& coords,
                   const std::vector<std::vector<std::uint8_t>>& candidates, DecoderCache::Layer& cache,
                   std::int64_t first_pos) const;
  std::vector<float> logits(const Tensor& hidden_last) const;

  DecoderDims dims_;
  RouterParams router_;
  PageRope rope_;
  Tensor embed_tokens_, final_norm_, lm_head_;
  std::vector<Layer> layers_;
};

// Candidate blocks for a query on image page `page` (-1 = none): the key
// blocks whose image tokens include that page. Empty when there is none,
// which routes over every block.
std::vector<std::uint8_t> page_candidates(const BlockPages& blocks, std::int64_t page, std::int64_t nblk);

}  // namespace uocr
