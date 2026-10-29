#include "weights/architecture.h"

#include <algorithm>
#include <format>
#include <set>
#include <utility>

#include "weights/sharded_weights.h"

namespace uocr {
namespace {

using Shape = std::vector<std::int64_t>;

class TableBuilder {
 public:
  void add(std::string name, Shape shape, DType dtype = DType::kBF16) {
    specs_.push_back({std::move(name), dtype, std::move(shape)});
  }
  void linear(const std::string& prefix, std::int64_t out, std::int64_t in, bool bias) {
    add(prefix + ".weight", {out, in});
    if (bias) {
      add(prefix + ".bias", {out});
    }
  }
  void norm(const std::string& prefix, std::int64_t width, bool bias) {
    add(prefix + ".weight", {width});
    if (bias) {
      add(prefix + ".bias", {width});
    }
  }
  std::vector<TensorSpec> take() { return std::move(specs_); }

 private:
  std::vector<TensorSpec> specs_;
};

void add_sam(TableBuilder& table, const SamDims& sam) {
  const std::string root = "model.sam_model";
  const std::int64_t grid = sam.image / sam.patch;  // 64
  const std::int64_t head_dim = sam.width / sam.heads;
  table.add(root + ".patch_embed.proj.weight", {sam.width, 3, sam.patch, sam.patch});
  table.add(root + ".patch_embed.proj.bias", {sam.width});
  table.add(root + ".pos_embed", {1, grid, grid, sam.width});
  for (std::int64_t layer = 0; layer < sam.layers; ++layer) {
    const std::string block = std::format("{}.blocks.{}", root, layer);
    const bool global = std::ranges::contains(sam.global_layers, layer);
    const std::int64_t span = global ? grid : sam.window;
    table.norm(block + ".norm1", sam.width, true);
    table.linear(block + ".attn.qkv", 3 * sam.width, sam.width, true);
    table.linear(block + ".attn.proj", sam.width, sam.width, true);
    table.add(block + ".attn.rel_pos_h", {2 * span - 1, head_dim});
    table.add(block + ".attn.rel_pos_w", {2 * span - 1, head_dim});
    table.norm(block + ".norm2", sam.width, true);
    table.linear(block + ".mlp.lin1", sam.mlp, sam.width, true);
    table.linear(block + ".mlp.lin2", sam.width, sam.mlp, true);
  }
  table.add(root + ".neck.0.weight", {sam.neck, sam.width, 1, 1});
  table.norm(root + ".neck.1", sam.neck, true);
  table.add(root + ".neck.2.weight", {sam.neck, sam.neck, 3, 3});
  table.norm(root + ".neck.3", sam.neck, true);
  table.add(root + ".net_2.weight", {sam.net2, sam.neck, 3, 3});
  table.add(root + ".net_3.weight", {sam.net3, sam.net2, 3, 3});
}

void add_clip(TableBuilder& table, const ClipDims& clip) {
  const std::string root = "model.vision_model";
  table.add(root + ".embeddings.class_embedding", {clip.width});
  table.add(root + ".embeddings.patch_embedding.weight", {clip.width, 3, clip.patch, clip.patch});
  table.add(root + ".embeddings.position_embedding.weight", {clip.positions, clip.width});
  table.add(root + ".embeddings.position_ids", {1, clip.positions}, DType::kI64);
  table.norm(root + ".pre_layrnorm", clip.width, true);  // upstream's spelling
  for (std::int64_t layer = 0; layer < clip.layers; ++layer) {
    const std::string block = std::format("{}.transformer.layers.{}", root, layer);
    table.norm(block + ".layer_norm1", clip.width, true);
    table.linear(block + ".self_attn.qkv_proj", 3 * clip.width, clip.width, true);
    table.linear(block + ".self_attn.out_proj", clip.width, clip.width, true);
    table.norm(block + ".layer_norm2", clip.width, true);
    table.linear(block + ".mlp.fc1", clip.mlp, clip.width, true);
    table.linear(block + ".mlp.fc2", clip.width, clip.mlp, true);
  }
}

void add_mlp(TableBuilder& table, const std::string& prefix, std::int64_t hidden, std::int64_t ffn) {
  table.linear(prefix + ".gate_proj", ffn, hidden, false);
  table.linear(prefix + ".up_proj", ffn, hidden, false);
  table.linear(prefix + ".down_proj", hidden, ffn, false);
}

void add_decoder(TableBuilder& table, const DecoderDims& dec) {
  const std::int64_t attn = dec.heads * dec.head_dim;
  table.add("model.embed_tokens.weight", {dec.vocab, dec.hidden});
  for (std::int64_t layer = 0; layer < dec.layers; ++layer) {
    const std::string block = std::format("model.layers.{}", layer);
    table.norm(block + ".input_layernorm", dec.hidden, false);
    table.linear(block + ".self_attn.q_proj", attn, dec.hidden, false);
    table.linear(block + ".self_attn.k_proj", attn, dec.hidden, false);
    table.linear(block + ".self_attn.v_proj", attn, dec.hidden, false);
    table.linear(block + ".self_attn.o_proj", dec.hidden, attn, false);
    for (const char* scorer : {"indexer", "dash_router"}) {
      const std::string head = block + ".self_attn." + scorer;
      table.linear(head + ".q_proj", dec.index_dim, dec.hidden, false);
      table.linear(head + ".k_proj", dec.index_dim, dec.hidden, false);
      table.norm(head + ".q_norm", dec.index_dim, false);
      table.norm(head + ".k_norm", dec.index_dim, false);
    }
    table.add(block + ".self_attn.dash_router.head_cls", {dec.index_dim});
    table.norm(block + ".post_attention_layernorm", dec.hidden, false);
    if (layer < dec.first_moe_layer) {
      add_mlp(table, block + ".mlp", dec.hidden, dec.dense_ffn);
      continue;
    }
    table.add(block + ".mlp.gate.weight", {dec.routed_experts, dec.hidden});
    for (std::int64_t expert = 0; expert < dec.routed_experts; ++expert) {
      add_mlp(table, std::format("{}.mlp.experts.{}", block, expert), dec.hidden, dec.expert_ffn);
    }
    add_mlp(table, block + ".mlp.shared_experts", dec.hidden, dec.shared_experts * dec.expert_ffn);
  }
  table.norm("model.norm", dec.hidden, false);
  table.add("lm_head.weight", {dec.vocab, dec.hidden});
}

std::string shape_text(const Shape& shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += std::format("{}{}", i == 0 ? "" : ", ", shape[i]);
  }
  return text + "]";
}

}  // namespace

std::vector<TensorSpec> unlimited_ocr2_tensors(const ModelDims& dims) {
  TableBuilder table;
  add_sam(table, dims.sam);
  add_clip(table, dims.clip);
  table.linear("model.projector.layers", dims.decoder.hidden, dims.projector_in, true);
  table.add("model.image_newline", {dims.decoder.hidden});
  table.add("model.view_seperator", {dims.decoder.hidden});  // upstream's spelling
  add_decoder(table, dims.decoder);
  return table.take();
}

std::vector<std::string> check_architecture(const std::vector<TensorSpec>& specs, const ShardedWeights& weights) {
  std::vector<std::string> problems;
  std::set<std::string> expected;
  for (const auto& spec : specs) {
    expected.insert(spec.name);
    const TensorInfo* info = weights.find(spec.name);
    if (info == nullptr) {
      problems.push_back(std::format("missing {}", spec.name));
      continue;
    }
    if (info->dtype != spec.dtype) {
      problems.push_back(std::format("{}: dtype {}, expected {}", spec.name, dtype_name(info->dtype),
                                     dtype_name(spec.dtype)));
    }
    if (info->shape != spec.shape) {
      problems.push_back(std::format("{}: shape {}, expected {}", spec.name, shape_text(info->shape),
                                     shape_text(spec.shape)));
    }
  }
  for (const auto& name : weights.names()) {
    if (!expected.contains(name)) {
      problems.push_back(std::format("unexpected {}", name));
    }
  }
  return problems;
}

}  // namespace uocr
