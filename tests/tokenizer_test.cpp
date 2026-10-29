// The real tokenizer: special ids the engine hard-codes, and the pinned
// instruction ids decoding to the instruction. Needs UOCR_MODELS_DIR.

#include "check.h"
#include "runtime/prompt_builder.h"
#include "tokenizer/tokenizer.h"

using namespace uocr;

int main() {
  const char* models = env_or_null("UOCR_MODELS_DIR");
  if (models == nullptr) return skip("UOCR_MODELS_DIR not set");
  return run_tests([&] {
    const auto tok = Tokenizer::load(std::filesystem::path(models) / "tokenizer.json");
    const PromptTokens t;
    require(tok.special_id("<PAGE>") == t.page && tok.special_id("<image>") == t.image, "page and image ids");
    require(tok.decode(t.instruction) == "Multi page parsing.", "instruction ids decode to the instruction");
    require(tok.decode({t.page, tok.special_id("<BLOCK>")}) == "<PAGE><BLOCK>", "special tokens decode literally");
    require(tok.special_id("<｜end▁of▁sentence｜>") == 1 && tok.special_id("<｜begin▁of▁sentence｜>") == t.bos, "bos and eos");
  });
}
