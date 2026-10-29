#pragma once

// Where model code gets its parameters from: the checkpoint in production,
// a small in-memory table in tests. Parameters come back as fp32 tensors
// for the reference backend.

#include <map>
#include <string>

#include "tensor/tensor.h"

namespace uocr {

class ShardedWeights;

class WeightSource {
 public:
  virtual ~WeightSource() = default;
  // Throws std::out_of_range naming the tensor when it is absent.
  virtual Tensor get(const std::string& name) const = 0;
};

// Reads from a checkpoint, widening bf16 to fp32.
class CheckpointWeights final : public WeightSource {
 public:
  explicit CheckpointWeights(const ShardedWeights& weights) : weights_(weights) {}
  Tensor get(const std::string& name) const override;

 private:
  const ShardedWeights& weights_;
};

// An in-memory table, for tests.
class TableWeights final : public WeightSource {
 public:
  void put(const std::string& name, Tensor tensor) { table_[name] = std::move(tensor); }
  Tensor get(const std::string& name) const override;

 private:
  std::map<std::string, Tensor> table_;
};

}  // namespace uocr
