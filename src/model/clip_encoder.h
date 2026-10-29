#pragma once

// The CLIP-L transformer, fed SAM's features as its patch embeddings (this
// model never runs CLIP's own patch convolution). SAM features
// [1024, g, g] become [1 + g * g, 1024]: a class token, then one token per
// SAM cell in row order.

#include <vector>

#include "model/weight_source.h"
#include "tensor/tensor.h"
#include "weights/architecture.h"

namespace uocr {

class ClipEncoder {
 public:
  ClipEncoder(const WeightSource& weights, const ClipDims& dims = {});

  Tensor forward(const Tensor& sam_features) const;

 private:
  struct Layer {
    Tensor ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b;
  };

  Tensor positions(std::int64_t grid) const;

  ClipDims dims_;
  Tensor class_embedding_, position_embedding_, pre_ln_w_, pre_ln_b_;
  std::vector<Layer> layers_;
};

}  // namespace uocr
