#pragma once

// SAM ViT-B image encoder with the two stride-2 convolutions this model
// adds after the neck. A [3, S, S] image (S a multiple of 64) becomes
// [1024, S/64, S/64] features: 16 x 16 for the 1024 global view, 10 x 10
// for a 640 tile.

#include <vector>

#include "model/weight_source.h"
#include "tensor/tensor.h"
#include "weights/architecture.h"

namespace uocr {

class SamEncoder {
 public:
  SamEncoder(const WeightSource& weights, const SamDims& dims = {});

  Tensor forward(const Tensor& image) const;

 private:
  struct Block {
    Tensor norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b;
    Tensor rel_pos_h, rel_pos_w;
    Tensor norm2_w, norm2_b, lin1_w, lin1_b, lin2_w, lin2_b;
    bool global = false;
  };

  Tensor block_forward(const Block& block, Tensor x) const;
  Tensor attention(const Block& block, const Tensor& x, std::int64_t h, std::int64_t w) const;

  SamDims dims_;
  Tensor patch_w_, patch_b_, pos_embed_;
  std::vector<Block> blocks_;
  Tensor neck0_w_, neck1_w_, neck1_b_, neck2_w_, neck3_w_, neck3_b_, net2_w_, net3_w_;
};

// The relative-position table for a query/key span of `size`, from a
// learned table of any length (resized linearly when it differs from
// 2 * size - 1): [size, size, head_dim], entry [q][k] = table[q - k + size - 1].
Tensor relative_positions(const Tensor& table, std::int64_t size);

}  // namespace uocr
