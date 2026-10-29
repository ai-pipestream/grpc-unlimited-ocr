#include "weights/sharded_weights.h"

#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace uocr {

ShardedWeights ShardedWeights::open(const std::filesystem::path& index_path) {
  std::ifstream stream(index_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(std::format("{}: cannot open the shard index", index_path.string()));
  }
  const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  const auto index = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (index.is_discarded() || !index.is_object() || !index.contains("weight_map") ||
      !index["weight_map"].is_object()) {
    throw std::runtime_error(std::format("{}: not a shard index (no weight_map object)", index_path.string()));
  }

  ShardedWeights weights;
  std::map<std::string, std::size_t> shard_slot;
  std::map<std::string, std::string> expected_shard;
  for (const auto& [name, shard] : index["weight_map"].items()) {
    if (!shard.is_string()) {
      throw std::runtime_error(std::format("{}: weight_map[{}] is not a file name", index_path.string(), name));
    }
    const auto file = shard.get<std::string>();
    if (file.find('/') != std::string::npos || file == ".." || file.empty()) {
      throw std::runtime_error(std::format("{}: shard name {} is not a plain file name", index_path.string(), file));
    }
    expected_shard.emplace(name, file);
    if (!shard_slot.contains(file)) {
      shard_slot.emplace(file, weights.shards_.size());
      weights.shards_.emplace_back(index_path.parent_path() / file);
    }
  }

  for (const auto& [file, slot] : shard_slot) {
    for (const auto& [name, info] : weights.shards_[slot].tensors()) {
      const auto expected = expected_shard.find(name);
      if (expected == expected_shard.end()) {
        throw std::runtime_error(std::format("{}: holds {}, which the index does not list", file, name));
      }
      if (expected->second != file) {
        throw std::runtime_error(std::format("{}: holds {}, which the index places in {}", file, name,
                                             expected->second));
      }
      weights.locations_.emplace(name, &info);
    }
  }
  for (const auto& [name, file] : expected_shard) {
    if (!weights.locations_.contains(name)) {
      throw std::runtime_error(std::format("{}: the index places {} here but the shard lacks it", file, name));
    }
  }
  return weights;
}

const TensorInfo* ShardedWeights::find(const std::string& name) const {
  const auto it = locations_.find(name);
  return it == locations_.end() ? nullptr : it->second;
}

const TensorInfo& ShardedWeights::at(const std::string& name) const {
  const TensorInfo* info = find(name);
  if (info == nullptr) {
    throw std::out_of_range(std::format("no tensor named {}", name));
  }
  return *info;
}

std::vector<std::string> ShardedWeights::names() const {
  std::vector<std::string> result;
  result.reserve(locations_.size());
  for (const auto& entry : locations_) {
    result.push_back(entry.first);
  }
  return result;
}

}  // namespace uocr
