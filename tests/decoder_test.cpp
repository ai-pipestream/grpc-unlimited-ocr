// The decoder on a tiny random model: decoding one token at a time gives
// the same logits as prefilling the longer sequence, generation is
// deterministic and respects the page guard.

#include <cmath>
#include <algorithm>
#include <random>

#include "check.h"
#include "model/decoder.h"
#include "runtime/generator.h"
#include "weights/architecture.h"

using namespace uocr;

namespace {

DecoderDims tiny() {
  DecoderDims d;
  d.vocab = 128830;  // keeps the real special ids addressable
  d.hidden = 32;
  d.layers = 2;
  d.heads = 2;
  d.head_dim = 16;
  d.dense_ffn = 48;
  d.routed_experts = 8;
  d.experts_per_token = 3;
  d.shared_experts = 1;
  d.expert_ffn = 24;
  d.index_dim = 16;
  return d;
}

TableWeights weights_for(const DecoderDims& d) {
  ModelDims md;
  md.decoder = d;
  std::mt19937 rng(3);
  std::normal_distribution<float> normal(0.0f, 0.2f);
  TableWeights table;
  for (const auto& spec : unlimited_ocr2_tensors(md)) {
    if (!spec.name.starts_with("model.layers") && !spec.name.starts_with("model.embed") &&
        spec.name != "model.norm.weight" && spec.name != "lm_head.weight") {
      continue;
    }
    Tensor t(spec.shape);
    const bool norm = spec.name.ends_with("norm.weight") && spec.name.find("router") == std::string::npos;
    for (float& v : t.span()) v = norm ? 1.0f : normal(rng);
    table.put(spec.name, std::move(t));
  }
  return table;
}

float max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

int main() {
  return run_tests([] {
    const DecoderDims d = tiny();
    const auto w = weights_for(d);
    RouterParams rp = Decoder::default_router();
    rp.block_size = 4;
    rp.local_blocks = 1;
    rp.idx_scale = 4.0f;
    const Decoder decoder(w, d, rp);

    // Two pages of 12 "image" tokens each, long enough that remote blocks exist.
    auto prompt = build_prompt({12, 12});
    const Tensor embeds = decoder.embed(prompt);
    const PromptLayout layout = layout_prompt(prompt);
    DecoderCache cache;
    const auto first = decoder.prefill(embeds, layout, cache);
    for (const float v : first) require(std::isfinite(v), "finite prefill logits");

    // Feed 3 text tokens by decode, then compare with prefilling them.
    const std::vector<std::int64_t> extra{7, 9, 11};
    DecodeState state(layout, prompt);
    const BlockPages blocks = block_image_pages(layout, rp.block_size);
    std::vector<float> stepped;
    for (const auto t : extra) {
      const RopeCoord c = state.advance(t);
      stepped = decoder.step(t, c, state.query_page(), blocks, cache);
    }
    auto longer = prompt;
    longer.insert(longer.end(), extra.begin(), extra.end());
    DecoderCache fresh;
    const auto whole = decoder.prefill(decoder.embed(longer), layout_prompt(longer), fresh);
    const float diff = max_diff(stepped, whole);
    require(diff < 1e-3f, std::format("decode matches prefill (max diff {})", diff));

    GenerationConfig gc;
    gc.max_new_tokens = 12;
    gc.max_page_tokens = 4;
    const auto a = generate_from_embeddings(decoder, prompt, embeds, gc);
    const auto b = generate_from_embeddings(decoder, prompt, embeds, gc);
    require(a.tokens == b.tokens, "greedy generation is deterministic");
    const std::size_t eos = a.finish == FinishReason::kEndOfSequence ? 1 : 0;
    require(a.trace.size() == a.tokens.size() + eos, "one trace row per step, EOS step included");
    for (std::size_t i = 0; i < a.tokens.size(); ++i) {
      const auto& s = a.trace[i];
      require(s.token == a.tokens[i] && s.logprob <= 0.0f && s.model_logprob >= s.logprob, "trace rows are consistent");
      require((s.intervention == Intervention::kNone) == (s.token == s.model_token), "an override is recorded when it happens");
    }
    require(std::ranges::count(a.tokens, PromptTokens{}.page) <= 2, "never more pages than the prompt has");
  });
}
