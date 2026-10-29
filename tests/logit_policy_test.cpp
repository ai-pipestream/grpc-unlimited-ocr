// The n-gram ban and the page-count guard.

#include <limits>

#include "check.h"
#include "runtime/logit_policy.h"

using namespace uocr;

int main() {
  return run_tests([] {
    LogitPolicyConfig cfg;
    cfg.page_token = 5;
    cfg.eos_token = 1;
    cfg.expected_pages = 2;
    cfg.max_page_tokens = 3;
    LogitPolicy guard(cfg);
    std::vector<float> logits(8, 0.0f);
    logits[1] = 9.0f;  // the model wants to stop
    require(guard.choose(logits, {5}).token != 1, "EOS banned before every page is out");
    std::vector<float> l2(8, 0.0f);
    l2[3] = 9.0f;
    require(guard.choose(l2, {5, 2, 2, 2}).token == 5, "a page at its cap is closed with <PAGE>");
    std::vector<float> l3(8, 0.0f);
    l3[3] = 9.0f;
    require(guard.choose(l3, {5, 2, 2, 2, 5, 2, 2, 2}).token == 1, "the last page at its cap ends the document");
    std::vector<float> l4(8, 0.0f);
    l4[5] = 9.0f;
    LogitPolicy over(cfg);
    require(over.choose(l4, {5, 5, 5}).token == 1, "too many markers forces EOS");

    LogitPolicyConfig ng;
    ng.ngram_size = 3;
    ng.ngram_window = 10;
    LogitPolicy ban(ng);
    std::vector<float> l5(8, 0.0f);
    l5[4] = 9.0f;
    l5[6] = 8.0f;
    // "2 3 4 ... 2 3" would repeat "2 3 4": 4 is banned, 6 wins.
    require(ban.choose(l5, {2, 3, 4, 7, 2, 3}).token == 6, "a repeated n-gram is banned");
    std::vector<float> l6(8, 0.0f);
    l6[4] = 9.0f;
    require(ban.choose(l6, {2, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 2, 3}).token == 4, "outside the window it is allowed");
  });
}
