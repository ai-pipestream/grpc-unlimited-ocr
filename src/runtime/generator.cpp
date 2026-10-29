#include "runtime/generator.h"

#include <algorithm>
#include <stdexcept>

namespace uocr {

GenerationResult generate(const VisionEncoder& vision, const Decoder& decoder, const std::vector<PageViews>& pages,
                          const GenerationConfig& config, const TokenSink& sink) {
  std::vector<Tensor> page_tokens;
  std::vector<std::int64_t> counts;
  for (const auto& page : pages) {
    page_tokens.push_back(vision.encode_page(page));
    counts.push_back(page_tokens.back().dim(0));
  }
  auto prompt = build_prompt(counts);
  Tensor embeds = decoder.embed(prompt);
  const auto positions = image_positions(prompt);
  std::size_t next = 0;
  for (const auto& tokens : page_tokens) {
    for (std::int64_t r = 0; r < tokens.dim(0); ++r) {
      std::copy_n(tokens.row(r), tokens.dim(1), embeds.row(positions[next++]));
    }
  }
  return generate_from_embeddings(decoder, std::move(prompt), embeds, config, sink);
}

GenerationResult generate_from_embeddings(const Decoder& decoder, std::vector<std::int64_t> prompt, const Tensor& embeds,
                                          const GenerationConfig& config, const TokenSink& sink) {
  const PromptTokens tokens;
  const PromptLayout layout = layout_prompt(prompt);
  const BlockPages blocks = block_image_pages(layout, Decoder::default_router().block_size);
  DecodeState state(layout, prompt);
  LogitPolicy policy(LogitPolicyConfig{
      config.ngram_size, config.ngram_window, {},
      config.page_count_guard ? static_cast<std::int64_t>(std::ranges::count(prompt, tokens.page)) : 0,
      config.max_page_tokens, tokens.page, 1});

  GenerationResult result;
  result.prompt = std::move(prompt);
  DecoderCache cache;
  std::vector<float> logits = decoder.prefill(embeds, layout, cache);
  const auto prompt_len = static_cast<std::int64_t>(result.prompt.size());
  for (std::int64_t step = 0; step < config.max_new_tokens; ++step) {
    const std::int64_t next = policy.choose(logits, result.tokens);
    if (next == 1) {
      result.finish = FinishReason::kEndOfSequence;
      return result;
    }
    result.tokens.push_back(next);
    if (sink && !sink(next)) {
      return result;
    }
    if (prompt_len + static_cast<std::int64_t>(result.tokens.size()) >= config.max_length) {
      result.finish = FinishReason::kContextLength;
      return result;
    }
    const RopeCoord coord = state.advance(next);
    logits = decoder.step(next, coord, state.query_page(), blocks, cache);
  }
  result.finish = FinishReason::kTokenBudget;
  return result;
}

}  // namespace uocr
