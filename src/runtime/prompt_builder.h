#pragma once

// The prompt the model was trained on: BOS, then "<PAGE>" and the page's
// image tokens for every page, then the instruction "Multi page parsing.".

#include <cstdint>
#include <vector>

namespace uocr {

struct PromptTokens {
  std::int64_t bos = 0;
  std::int64_t page = 128827;
  std::int64_t image = 128815;
  // "Multi page parsing." under the model's tokenizer: Multi, Ġpage,
  // Ġparsing, "." (tokenizer_test checks they decode back to the text).
  std::vector<std::int64_t> instruction = {37460, 4366, 76466, 16};
};

// `image_tokens[i]` is page i's image-token count (273 in base mode).
std::vector<std::int64_t> build_prompt(const std::vector<std::int64_t>& image_tokens, const PromptTokens& tokens = {});

// Indices of the image-token positions, in order: where the vision
// encoder's rows are written.
std::vector<std::int64_t> image_positions(const std::vector<std::int64_t>& prompt, const PromptTokens& tokens = {});

}  // namespace uocr
