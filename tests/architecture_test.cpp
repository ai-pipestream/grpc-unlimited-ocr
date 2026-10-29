// The architecture table: its size and a few anchors, without weights.

#include <set>

#include "check.h"
#include "weights/architecture.h"

using namespace uocr;

int main() {
  return run_tests([] {
    const auto specs = unlimited_ocr2_tensors();
    std::set<std::string> names;
    for (const auto& spec : specs) {
      require(names.insert(spec.name).second, "duplicate name " + spec.name);
    }
    // The published checkpoint's index lists 2,819 tensors.
    require(specs.size() == 2819, std::format("table has {} tensors, the checkpoint 2819", specs.size()));
    auto find = [&](const std::string& name) -> const TensorSpec& {
      for (const auto& spec : specs) {
        if (spec.name == name) return spec;
      }
      throw std::runtime_error("FAIL: no " + name);
    };
    require(find("model.sam_model.blocks.5.attn.rel_pos_h").shape == std::vector<std::int64_t>{127, 64},
            "global SAM block spans the whole grid");
    require(find("model.sam_model.blocks.4.attn.rel_pos_w").shape == std::vector<std::int64_t>{27, 64},
            "windowed SAM block spans 14");
    require(find("model.layers.0.mlp.gate_proj.weight").shape == std::vector<std::int64_t>{6848, 1280},
            "layer 0 is dense");
    require(find("model.layers.11.mlp.experts.63.down_proj.weight").shape == std::vector<std::int64_t>{1280, 896},
            "last expert");
    require(find("model.vision_model.embeddings.position_ids").dtype == DType::kI64, "position ids are I64");
  });
}
