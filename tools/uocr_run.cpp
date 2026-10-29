// Developer runner: parses page images with the reference engine and
// streams the model's raw output to stdout. Not the service (that one is
// diskless and speaks gRPC); this reads its pages from files.
//
//   uocr-run --models DIR [--tiled] [--ngram N --window N] [--max-new N] page1.png [page2.png ...]

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <print>
#include <string>
#include <vector>

#include "image/page_preprocess.h"
#include "model/decoder.h"
#include "model/vision_encoder.h"
#include "runtime/generator.h"
#include "tokenizer/tokenizer.h"
#include "weights/architecture.h"
#include "weights/sharded_weights.h"

using namespace uocr;

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int usage() {
  std::println(stderr, "usage: uocr-run --models DIR [--tiled] [--ngram N] [--window N] [--max-new N] page...");
  return 64;
}

}  // namespace

int main(int argc, char** argv) {
  std::string models;
  std::vector<std::string> pages;
  PreprocessOptions pre;
  GenerationConfig gen;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
      return argv[++i];
    };
    if (a == "--models") models = value();
    else if (a == "--tiled") pre.tiled = true;
    else if (a == "--ngram") gen.ngram_size = std::stoll(value());
    else if (a == "--window") gen.ngram_window = std::stoll(value());
    else if (a == "--max-new") gen.max_new_tokens = std::stoll(value());
    else if (a.starts_with("--")) return usage();
    else pages.push_back(a);
  }
  if (models.empty() || pages.empty()) return usage();
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const auto shards = ShardedWeights::open(std::filesystem::path(models) / "model.safetensors.index.json");
    const auto problems = check_architecture(unlimited_ocr2_tensors(), shards);
    if (!problems.empty()) {
      for (const auto& p : problems) std::println(stderr, "{}", p);
      return 1;
    }
    const Tokenizer tokenizer = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const CheckpointWeights weights(shards);
    const VisionEncoder vision(weights);
    const Decoder decoder(weights);
    std::vector<PageViews> views;
    for (const auto& p : pages) views.push_back(preprocess_page(decode_image(read_file(p)), pre));
    const auto t1 = std::chrono::steady_clock::now();
    std::println(stderr, "loaded in {:.1f} s", std::chrono::duration<double>(t1 - t0).count());

    std::vector<std::int64_t> so_far;
    std::size_t printed = 0;
    auto first = std::chrono::steady_clock::time_point{};
    const auto result = generate(vision, decoder, views, gen, [&](std::int64_t token) {
      if (so_far.empty()) first = std::chrono::steady_clock::now();
      so_far.push_back(token);
      const std::string text = tokenizer.decode(so_far);
      std::print("{}", text.substr(printed));
      std::fflush(stdout);
      printed = text.size();
      return true;
    });
    const auto t2 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t2 - first).count();
    std::println(stderr, "\nprompt {} tokens, prefill {:.1f} s, {} tokens in {:.1f} s ({:.2f} tok/s), finish {}",
                 result.prompt.size(), std::chrono::duration<double>(first - t1).count(), result.tokens.size(),
                 decode_s, static_cast<double>(result.tokens.size()) / std::max(decode_s, 1e-9),
                 static_cast<int>(result.finish));
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
  return 0;
}
