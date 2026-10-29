#include "runtime/prompt_builder.h"

namespace uocr {

std::vector<std::int64_t> build_prompt(const std::vector<std::int64_t>& image_tokens, const PromptTokens& tokens) {
  std::vector<std::int64_t> ids{tokens.bos};
  for (const std::int64_t count : image_tokens) {
    ids.push_back(tokens.page);
    ids.insert(ids.end(), static_cast<std::size_t>(count), tokens.image);
  }
  ids.insert(ids.end(), tokens.instruction.begin(), tokens.instruction.end());
  return ids;
}

std::vector<std::int64_t> image_positions(const std::vector<std::int64_t>& prompt, const PromptTokens& tokens) {
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (prompt[i] == tokens.image) out.push_back(static_cast<std::int64_t>(i));
  }
  return out;
}

}  // namespace uocr
