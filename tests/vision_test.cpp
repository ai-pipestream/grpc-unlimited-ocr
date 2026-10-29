// The vision encoder end to end on a tiny, randomly initialised model:
// token layout in base and tiled mode, and bitwise determinism across
// thread counts. The real weights run in vision_checkpoint_test.

#include <cmath>
#include <random>

#include "backend/reference/parallel.h"
#include "check.h"
#include "model/vision_encoder.h"
#include "weights/architecture.h"

using namespace uocr;

namespace {

ModelDims tiny_dims() {
  ModelDims d;
  d.sam.width = 24;
  d.sam.layers = 2;
  d.sam.heads = 2;
  d.sam.mlp = 48;
  d.sam.global_layers = {1, 99, 99, 99};
  d.sam.neck = 8;
  d.sam.net2 = 12;
  d.sam.net3 = 16;
  d.clip.width = 16;
  d.clip.layers = 2;
  d.clip.heads = 2;
  d.clip.mlp = 32;
  d.projector_in = 32;
  d.decoder.hidden = 20;
  return d;
}

TableWeights random_weights(const ModelDims& dims) {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0f, 0.05f);
  TableWeights table;
  for (const auto& spec : unlimited_ocr2_tensors(dims)) {
    if (!spec.name.starts_with("model.sam_model") && !spec.name.starts_with("model.vision_model") &&
        !spec.name.starts_with("model.projector") && spec.name != "model.image_newline" &&
        spec.name != "model.view_seperator") {
      continue;
    }
    Tensor t(spec.shape);
    const bool is_norm_weight = spec.name.find("norm") != std::string::npos && spec.name.ends_with(".weight");
    for (float& v : t.span()) v = is_norm_weight ? 1.0f + normal(rng) : normal(rng);
    table.put(spec.name, std::move(t));
  }
  return table;
}

Tensor random_image(std::int64_t size, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  Tensor img({3, size, size});
  for (float& v : img.span()) v = unit(rng);
  return img;
}

}  // namespace

int main() {
  return run_tests([] {
    const auto dims = tiny_dims();
    const auto weights = random_weights(dims);
    const VisionEncoder vision(weights, dims);

    // 128 px views give a 2 x 2 feature grid.
    const Tensor view = vision.encode_view(random_image(128, 1));
    require(view.shape() == std::vector<std::int64_t>{4, 20}, "a view projects to [g*g, hidden]");

    PageViews base{random_image(128, 2), {}, 1, 1};
    const Tensor base_tokens = vision.encode_page(base);
    require(base_tokens.dim(0) == page_image_tokens(2, 2, 1, 1, false), "base layout size");
    require(base_tokens.dim(0) == 2 * 3 + 1, "base layout is g*(g+1)+1");

    PageViews tiled{random_image(128, 3), {random_image(128, 4), random_image(128, 5)}, 2, 1};
    const Tensor tiled_tokens = vision.encode_page(tiled);
    require(tiled_tokens.dim(0) == page_image_tokens(2, 2, 2, 1, true), "tiled layout size");

    // The last row is the separator and each row of the global grid ends in the newline.
    const Tensor newline = weights.get("model.image_newline");
    const Tensor separator = weights.get("model.view_seperator");
    const std::int64_t last = base_tokens.dim(0) - 1;
    for (std::int64_t e = 0; e < 20; ++e) {
      require(base_tokens.row(last)[e] == separator.data()[e], "separator closes the page");
      require(base_tokens.row(2)[e] == newline.data()[e] && base_tokens.row(5)[e] == newline.data()[e],
              "newline after each global row");
    }
    for (const float v : tiled_tokens.span()) require(std::isfinite(v), "finite tokens");

    reference::set_thread_count(1);
    const Tensor serial = vision.encode_page(tiled);
    reference::set_thread_count(std::thread::hardware_concurrency());
    for (std::int64_t i = 0; i < serial.numel(); ++i) {
      require(serial.data()[i] == tiled_tokens.data()[i], "bitwise identical across thread counts");
    }
  });
}
