// The body of one CUDA decode step, written so it can be captured as a
// graph: every buffer is fixed for the life of the capture, and every
// length-dependent value is read from the device step state.

#include <cmath>

#include "backend/cuda/decode_ops.h"
#include "backend/cuda/ops.h"
#include "model/cuda/cuda_decoder.h"

namespace uocr::cuda {

void CudaDecoder::decode_attention(const Layer& layer, Cache& cache) {
  const std::int64_t width = dims_.heads * dims_.head_dim, hd = dims_.head_dim, d = dims_.index_dim;
  (void)width;
  const auto* state = static_cast<const StepState*>(step_.state.data());
  const float* h = step_.h.f32();
  apply_weight(ctx_, h, 1, layer.q, nullptr, step_.q.f32());
  apply_weight(ctx_, h, 1, layer.k, nullptr, step_.k.f32());
  apply_weight(ctx_, h, 1, layer.v, nullptr, step_.v.f32());
  apply_weight(ctx_, h, 1, layer.router_q, nullptr, step_.rq.f32());
  apply_weight(ctx_, h, 1, layer.router_k, nullptr, step_.rk.f32());
  const float eps = static_cast<float>(dims_.rms_norm_eps);
  router_norm(ctx_, step_.rq.f32(), layer.router_q_norm.f32(), 1, d, eps);
  router_norm(ctx_, step_.rk.f32(), layer.router_k_norm.f32(), 1, d, eps);
  const auto n_tok = static_cast<std::int64_t>(rope_.inv_tok().size());
  rope_step(ctx_, step_.q.f32(), dims_.heads, hd, state, inv_tok_.f32(), n_tok, inv_page_.f32(), rope_.page_mod());
  rope_step(ctx_, step_.k.f32(), dims_.heads, hd, state, inv_tok_.f32(), n_tok, inv_page_.f32(), rope_.page_mod());
  rope_step(ctx_, step_.rq.f32(), 1, d, state, inv_tok_.f32(), n_tok, inv_page_.f32(), rope_.page_mod());
  rope_step(ctx_, step_.rk.f32(), 1, d, state, inv_tok_.f32(), n_tok, inv_page_.f32(), rope_.page_mod());
  append_step(ctx_, cache.k.f32(), step_.k.f32(), width, state);
  append_step(ctx_, cache.v.f32(), step_.v.f32(), width, state);
  append_step(ctx_, cache.router_k.f32(), step_.rk.f32(), d, state);

  summary_step(ctx_, cache.router_k.f32(), d, layer.head_cls.f32(), router_.block_size, state, cache.summaries.f32());
  scores_step(ctx_, step_.rq.f32(), cache.summaries.f32(), d, router_.block_size, state, capacity_blocks_,
              step_.z.f32());
  route_step(ctx_, step_.z.f32(), capacity_blocks_, block_min_.i64(), block_max_.i64(), prompt_blocks_, state,
             {router_.block_size, router_.local_blocks, router_.top_m, router_.top_k, router_.idx_scale, router_.floor,
              router_.sigma},
             step_.mask.f32());
  attend_step(ctx_, step_.q.f32(), cache.k.f32(), cache.v.f32(), dims_.heads, hd, step_.mask.f32(),
              1.0f / std::sqrt(static_cast<float>(hd)), state, step_.scores.f32(), capacity_, step_.merged.f32());
  apply_weight(ctx_, step_.merged.f32(), 1, layer.o, nullptr, step_.delta.f32());
}

void CudaDecoder::decode_body() {
  const std::int64_t hidden = dims_.hidden;
  const float eps = static_cast<float>(dims_.rms_norm_eps);
  const auto* state = static_cast<const StepState*>(step_.state.data());
  embed_step(ctx_, embed_.bf16(), state, hidden, step_.x.f32());
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    copy(ctx_, step_.h.f32(), step_.x.f32(), hidden);
    rms_norm(ctx_, step_.h.f32(), layer.input_norm.f32(), 1, hidden, eps);
    decode_attention(layer, cache_[li]);
    add_inplace(ctx_, step_.x.f32(), step_.delta.f32(), hidden);
    copy(ctx_, step_.h.f32(), step_.x.f32(), hidden);
    rms_norm(ctx_, step_.h.f32(), layer.post_norm.f32(), 1, hidden, eps);
    if (layer.dense) {
      mlp(*layer.dense, step_.h, 1, step_.delta);
    } else {
      moe(layer, step_.h, 1, step_.delta);
    }
    add_inplace(ctx_, step_.x.f32(), step_.delta.f32(), hidden);
  }
  copy(ctx_, step_.h.f32(), step_.x.f32(), hidden);
  rms_norm(ctx_, step_.h.f32(), final_norm_.f32(), 1, hidden, eps);
  apply_weight(ctx_, step_.h.f32(), 1, lm_head_, nullptr, step_.logits.f32());
}

}  // namespace uocr::cuda
