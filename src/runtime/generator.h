#pragma once

// One document through the model: encode the pages, build the prompt,
// prefill, then decode greedily until EOS or a cap, reporting each token as
// it is chosen.

#include <cstdint>
#include <functional>
#include <vector>

#include "model/decoder.h"
#include "model/vision_encoder.h"
#include "runtime/logit_policy.h"
#include "runtime/prompt_builder.h"

namespace uocr {

enum class FinishReason { kEndOfSequence, kTokenBudget, kContextLength };

struct GenerationConfig {
  std::int64_t max_new_tokens = 16384;
  std::int64_t max_length = 131072;
  std::int64_t ngram_size = 0;
  std::int64_t ngram_window = 100;
  bool page_count_guard = true;
  std::int64_t max_page_tokens = 20000;
};

struct GenerationResult {
  std::vector<std::int64_t> prompt;
  std::vector<std::int64_t> tokens;  // generated, EOS excluded
  FinishReason finish = FinishReason::kTokenBudget;
};

// Called with each generated token as soon as it is chosen; return false to
// stop early.
using TokenSink = std::function<bool(std::int64_t token)>;

GenerationResult generate(const VisionEncoder& vision, const Decoder& decoder, const std::vector<PageViews>& pages,
                          const GenerationConfig& config, const TokenSink& sink = {});

// The same loop from prompt embeddings already built (tests drive the
// decoder this way without a vision encoder).
GenerationResult generate_from_embeddings(const Decoder& decoder, std::vector<std::int64_t> prompt, const Tensor& embeds,
                                          const GenerationConfig& config, const TokenSink& sink = {});

}  // namespace uocr
