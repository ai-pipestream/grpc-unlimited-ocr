#include "runtime/generator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uocr {
namespace {

// log(sum(exp(logits))): the normaliser of a log-softmax, computed once
// per step for every log-probability the step records.
double log_normaliser(const std::vector<float>& logits) {
  const float peak = *std::ranges::max_element(logits);
  double total = 0.0;
  for (const float v : logits) total += std::exp(static_cast<double>(v - peak));
  return peak + std::log(total);
}

}  // namespace

GenerationResult generate(const VisionModel& vision, DecoderModel& decoder, const std::vector<PageViews>& pages,
                          const GenerationConfig& config, const TokenSink& sink, const PrefillSink& prefilled) {
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
  return generate_from_embeddings(decoder, std::move(prompt), embeds, config, sink, prefilled);
}

GenerationResult generate_from_embeddings(DecoderModel& decoder, std::vector<std::int64_t> prompt, const Tensor& embeds,
                                          const GenerationConfig& config, const TokenSink& sink, const PrefillSink& prefilled) {
  const PromptTokens tokens;
  const PromptLayout layout = layout_prompt(prompt);
  const BlockPages blocks = block_image_pages(layout, decoder.block_size());
  DecodeState state(layout, prompt);
  LogitPolicy policy(LogitPolicyConfig{
      config.ngram_size, config.ngram_window, {},
      config.page_count_guard ? static_cast<std::int64_t>(std::ranges::count(prompt, tokens.page)) : 0,
      config.max_page_tokens, tokens.page, 1});

  GenerationResult result;
  result.prompt = std::move(prompt);
  std::vector<float> logits = decoder.prefill(embeds, layout);
  if (prefilled) prefilled(result.prompt.size());
  const auto prompt_len = static_cast<std::int64_t>(result.prompt.size());
  for (std::int64_t step = 0; step < config.max_new_tokens; ++step) {
    const std::vector<float> raw = logits;
    const Choice choice = policy.choose(logits, result.tokens);
    const std::int64_t next = choice.token;
    const double norm = log_normaliser(raw);
    auto logprob = [&](std::int64_t t) { return static_cast<float>(raw[static_cast<std::size_t>(t)] - norm); };
    const GeneratedToken record{next, logprob(next), choice.model_token, logprob(choice.model_token),
                                choice.intervention};
    result.trace.push_back(record);
    const bool keep_going = !sink || sink(record);
    if (next == 1) {
      result.finish = FinishReason::kEndOfSequence;
      return result;
    }
    result.tokens.push_back(next);
    if (!keep_going) {
      return result;
    }
    if (prompt_len + static_cast<std::int64_t>(result.tokens.size()) >= config.max_length) {
      result.finish = FinishReason::kContextLength;
      return result;
    }
    const RopeCoord coord = state.advance(next);
    logits = decoder.step(next, coord, state.query_page(), blocks);
  }
  result.finish = FinishReason::kTokenBudget;
  return result;
}

}  // namespace uocr
