#pragma once

// What the generation loop needs from a decoder, whatever backend runs it:
// embeddings, one prefill per document, then one step per token. An
// implementation keeps its own cache, so one instance decodes one document
// at a time.

#include <cstdint>
#include <vector>

#include "model/page_layout.h"
#include "tensor/tensor.h"

namespace uocr {

class DecoderModel {
 public:
  virtual ~DecoderModel() = default;

  virtual Tensor embed(const std::vector<std::int64_t>& ids) const = 0;

  // Starts a document; returns the next-token logits after the prompt.
  virtual std::vector<float> prefill(const Tensor& embeds, const PromptLayout& layout) = 0;

  // Feeds one token; returns the next-token logits.
  virtual std::vector<float> step(std::int64_t token, RopeCoord coord, std::int64_t query_page,
                                  const BlockPages& prompt_blocks) = 0;

  // The key block size page bookkeeping is done in.
  virtual std::int64_t block_size() const = 0;
};

}  // namespace uocr
