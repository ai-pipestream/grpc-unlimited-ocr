// Block routing: entmax values, the score mask, selection and bias.

#include <cmath>

#include "check.h"
#include "model/block_router.h"

using namespace uocr;

namespace {
bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
}

int main() {
  return run_tests([] {
    std::vector<float> even{0.0f, 0.0f};
    entmax15(even);
    require(near(even[0], 0.5f) && near(even[1], 0.5f), "entmax of a tie is uniform");
    std::vector<float> peaked{10.0f, 0.0f, -3.0f};
    entmax15(peaked);
    require(near(peaked[0], 1.0f) && peaked[1] == 0.0f && peaked[2] == 0.0f, "entmax drops far scores to exactly zero");
    std::vector<float> mixed{1.0f, 0.5f, -5.0f};
    entmax15(mixed);
    require(near(mixed[0] + mixed[1] + mixed[2], 1.0f) && mixed[2] == 0.0f && mixed[0] > mixed[1], "entmax sums to one");

    // 10 keys in blocks of 2: query at position 9 (block 4) with 1 local
    // block may score blocks 0..3.
    RouterParams p;
    p.block_size = 2;
    p.local_blocks = 1;
    p.top_m = 3;
    p.top_k = 2;
    p.idx_scale = 1.0f;
    Tensor idx_k({10, 2}), head_cls({2}, 0.0f);
    for (std::int64_t t = 0; t < 10; ++t) {
      idx_k.row(t)[0] = static_cast<float>(t);
      idx_k.row(t)[1] = 1.0f;
    }
    const Tensor summaries = block_summaries(idx_k, head_cls, 2);
    require(summaries.dim(0) == 5 && near(summaries.row(1)[0], 2.5f), "a zero head_cls averages its block");
    const Tensor q({1, 2}, {1.0f, 0.0f});
    const Tensor s = block_scores(q, summaries, {9}, 10, p);
    require(s.row(0)[4] == kScoreMasked && s.row(0)[3] > s.row(0)[0], "local block masked, later blocks score higher");

    const BlockSelection all = select_blocks(s, {}, p);
    require(all.width == 2 && all.block[0] == 3 && all.block[1] == 2 && all.valid[0] && all.valid[1], "top blocks chosen");
    require(near(all.bias[0] + all.bias[1], 0.0f), "biases are centred on their mean");

    // Restrict the candidates to block 0: it is chosen first, the second slot is padding.
    const BlockSelection one = select_blocks(s, {1, 0, 0, 0, 0}, p);
    require(one.block[0] == 0 && one.valid[0] && !one.valid[1] && one.bias[1] == 0.0f, "candidate restriction");
    // No candidate among scoreable blocks: fall back to all of them.
    const BlockSelection fallback = select_blocks(s, {0, 0, 0, 0, 1}, p);
    require(fallback.block[0] == 3 && fallback.valid[0], "empty candidate set falls back");
  });
}
