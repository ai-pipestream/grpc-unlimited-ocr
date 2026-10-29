#pragma once

// The decoder's feed-forward blocks: a gated SiLU MLP, and the mixture of
// experts that replaces it from layer 1 on (64 routed experts, 6 chosen per
// token by a softmax gate, plus 2 shared experts always applied).

#include <optional>
#include <string>
#include <vector>

#include "model/weight_source.h"
#include "tensor/tensor.h"
#include "weights/architecture.h"

namespace uocr {

class GatedMlp {
 public:
  GatedMlp(const WeightSource& weights, const std::string& prefix);
  Tensor forward(const Tensor& x) const;  // x [L, hidden]

 private:
  Tensor gate_, up_, down_;
};

class MixtureOfExperts {
 public:
  MixtureOfExperts(const WeightSource& weights, const std::string& prefix, const DecoderDims& dims);
  Tensor forward(const Tensor& x) const;  // x [L, hidden]

 private:
  Tensor gate_;  // [experts, hidden]
  std::vector<GatedMlp> experts_;
  GatedMlp shared_;
  std::int64_t top_k_;
};

class FeedForward {
 public:
  FeedForward(const WeightSource& weights, const std::string& prefix, const DecoderDims& dims, bool dense);
  Tensor forward(const Tensor& x) const;

 private:
  std::optional<GatedMlp> dense_;
  std::optional<MixtureOfExperts> moe_;
};

}  // namespace uocr
