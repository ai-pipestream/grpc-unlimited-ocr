#pragma once

// A checkpoint split across safetensors shards, addressed by tensor name
// through the shard index (`model.safetensors.index.json`).

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "weights/safetensors.h"

namespace uocr {

class ShardedWeights {
 public:
  // Opens every shard the index names, relative to the index's directory,
  // and checks that each indexed tensor is in the shard the index says and
  // that no shard holds a tensor the index does not list. Throws
  // std::runtime_error naming the first problem.
  static ShardedWeights open(const std::filesystem::path& index_path);

  // Nullptr when no shard holds the name.
  const TensorInfo* find(const std::string& name) const;

  // Throws std::out_of_range naming the tensor when it is absent.
  const TensorInfo& at(const std::string& name) const;

  std::size_t tensor_count() const { return locations_.size(); }

  // Every tensor name, sorted.
  std::vector<std::string> names() const;

 private:
  std::vector<SafetensorsFile> shards_;
  std::map<std::string, const TensorInfo*> locations_;
};

}  // namespace uocr
