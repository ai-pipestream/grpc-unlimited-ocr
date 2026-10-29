#pragma once

// Page bookkeeping over the token sequence, the part of sparse attention
// that is control flow rather than tensor math. From the prompt's token
// ids it derives each position's rope coordinate and the page each image
// token and each query belongs to; during decode it advances those as
// tokens are generated. Prompts look like "<PAGE><image>...<PAGE><image>...
// instruction", and the response repeats one <PAGE> per page.

#include <cstdint>
#include <vector>

#include "model/page_rope.h"

namespace uocr {

struct PageTokens {
  std::int64_t image = 128815;
  std::int64_t page = 128827;
};

// What the prompt fixes, computed once before prefill.
struct PromptLayout {
  std::vector<RopeCoord> coords;           // per prompt position
  std::vector<std::int64_t> image_page;    // per position: page of an image token, else -1
  std::vector<std::int64_t> query_page;    // per position: page a query belongs to, -1 = global
  std::vector<std::int64_t> prefill_count; // per rope page bucket: prompt positions in it
  std::int64_t rope_pages = 0;             // buckets, including bucket 0 (no page)
  std::int64_t prompt_pages = 0;           // <PAGE> markers up to the last image token
};

// Requires at least one image token and one <PAGE> before the last image;
// throws std::invalid_argument otherwise.
PromptLayout layout_prompt(const std::vector<std::int64_t>& ids, const PageTokens& tokens = {});

// Per key block of `block_size` prompt positions, the lowest and highest
// image page it holds (-1, -1 when it holds no image token).
struct BlockPages {
  std::vector<std::int64_t> min_page;
  std::vector<std::int64_t> max_page;
};
BlockPages block_image_pages(const PromptLayout& layout, std::int64_t block_size);

// The decode-time state: the rope coordinate of the next fed token and the
// response page the query is on.
class DecodeState {
 public:
  DecodeState(const PromptLayout& layout, const std::vector<std::int64_t>& prompt_ids, const PageTokens& tokens = {});

  // Feeds one generated token; returns its rope coordinate.
  RopeCoord advance(std::int64_t token);

  // Response page the last fed token is on (0-based, image page k <->
  // response page k), or -1 before the first response <PAGE>.
  std::int64_t query_page() const;

 private:
  const PromptLayout& layout_;
  PageTokens tokens_;
  RopeCoord coord_;
  std::int64_t markers_ = 0;  // <PAGE> tokens seen, prompt and response
};

}  // namespace uocr
