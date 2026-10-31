// The block grammar and the incremental parser: the golden page's output,
// then each rule ported from upstream's process_unlimitedOCR2.py.

#include "check.h"
#include "output/html_table.h"
#include "output_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

std::string golden_page() {
  std::string raw = read_text(std::string(UOCR_GOLDENS) + "/quickstart-p2/output_raw.txt");
  require(raw.starts_with("<PAGE>"), "golden output starts with <PAGE>");
  return raw.substr(6);
}

const v1::Block& only(const std::vector<v1::Block>& blocks) {
  require(blocks.size() == 1, "one block, got " + std::to_string(blocks.size()));
  return blocks.front();
}

void golden() {
  const std::string page = golden_page();
  const auto blocks = parse_text(page);
  require(blocks.size() == 6, "six blocks on the golden page");
  const v1::BlockKind kinds[] = {v1::BLOCK_KIND_TITLE, v1::BLOCK_KIND_TEXT,  v1::BLOCK_KIND_CHART,
                                 v1::BLOCK_KIND_IMAGE_CAPTION, v1::BLOCK_KIND_IMAGE, v1::BLOCK_KIND_IMAGE_CAPTION};
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    require(blocks[i].kind() == kinds[i], "kind of block " + std::to_string(i));
    require(blocks[i].ordinal() == i, "ordinal of block " + std::to_string(i));
  }
  require(blocks[0].text().text() == "Scanned page: OCR, figures, barcode", "title text");
  require(!blocks[0].has_heading_level(), "a title without <LEVEL> has no level");
  require(blocks[0].box().x1() == 90 && blocks[0].box().y1() == 67 && blocks[0].box().x2() == 824 &&
              blocks[0].box().y2() == 99,
          "title box");
  require(blocks[2].has_picture() && blocks[2].picture().text().empty(), "an empty chart");
  require(blocks[5].text().text().starts_with("Figure 2: the QR payload"), "last caption text");
  for (const unsigned seed : {1u, 2u, 3u, 9u}) {
    require(same_blocks(parse_text(page, seed), parse_whole(page)), "one token at a time equals all at once");
  }
  const auto one_byte = parse_text(page, 5, 1);
  require(same_blocks(one_byte, blocks), "one byte at a time");
}

void level_and_box_order() {
  const auto a = only(parse_whole("<BLOCK>title<LEVEL2>[1,2,3,4]<FORK>Intro"));
  require(a.kind() == v1::BLOCK_KIND_TITLE && a.heading_level() == 2 && a.box().x2() == 3, "level first");
  require(a.text().text() == "Intro", "level-first text");
  const auto b = only(parse_whole("<BLOCK>title[5,6,7,8]<LEVEL3><FORK>Next"));
  require(b.heading_level() == 3 && b.box().x1() == 5 && b.text().text() == "Next", "box first");
  const auto c = only(parse_whole("<BLOCK>text[1\xEF\xBC\x8C 2,3 ,4]<FORK>wide comma"));
  require(c.has_box() && c.box().y1() == 2 && c.box().y2() == 4, "full-width commas");
  const auto d = only(parse_whole("<BLOCK>text<FORK>no box"));
  require(!d.has_box() && d.raw_box_size() == 0 && d.text().text() == "no box", "a block without a box");
}

void raw_boxes_kept() {
  const auto b = only(parse_whole("<BLOCK>image[-5,20,1200,40,7]<FORK>"));
  require(b.box().x1() == 0 && b.box().x2() == 1000, "coordinates clamped to 0..1000");
  require(b.raw_box_size() == 5 && b.raw_box(0) == -5 && b.raw_box(2) == 1200 && b.raw_box(4) == 7,
          "every coordinate as generated");
}

void bare_labels() {
  const auto blocks = parse_whole("<BLOCK>text[1,1,2,2]<FORK>first line\ntext[3,3,4,4]second\nfoo[1,2,3,4]kept");
  require(blocks.size() == 2, "a known bare label starts a block, an unknown one does not");
  require(blocks[1].text().text() == "second\nfoo[1,2,3,4]kept", "unknown bare label stays as text");
  require(blocks[1].box().x1() == 3, "bare block box");
  const auto det = parse_whole("<FORK><|det|>table[1,2,3,4]<table><tr><td>a</td></tr></table>");
  require(det.size() == 1 && det[0].kind() == v1::BLOCK_KIND_TABLE, "a bare label behind <|det|> at the start");
  require(det[0].table().cells_size() == 1 && det[0].table().cells(0).text() == "a", "table cells");
  require(det[0].table().source_html() == "<table><tr><td>a</td></tr></table>", "source HTML verbatim");
}

void unrecognised_text() {
  const auto blocks = parse_whole("stray words <BLOCK>text[1,2,3,4]<FORK>body");
  require(blocks.size() == 2, "text before the first block is its own block");
  require(blocks[0].kind() == v1::BLOCK_KIND_UNRECOGNISED && blocks[0].text().text() == "stray words ",
          "unrecognised text verbatim");
  const auto quiet = parse_pieces({"\n ", "<BLOCK>", "text[1,2,3,4]", "<FORK>", "body"});
  require(quiet.size() == 1 && quiet[0].first_step() == 0, "whitespace before the first block joins it");
  output::BlockParser parser(1);
  const auto none = parse_pieces({"just ", "words"}, 1, &parser);
  require(none.size() == 1 && none[0].kind() == v1::BLOCK_KIND_UNRECOGNISED, "a page with no blocks");
  require(none[0].text().text() == "just words", "kept verbatim");
  require(parser.unparseable(), "and reported unparseable");
  const auto unknown = only(parse_whole("<BLOCK>foo[1,2,3,4]<FORK>odd"));
  require(unknown.kind() == v1::BLOCK_KIND_UNRECOGNISED && unknown.label_text() == "foo", "unknown label kept");
  require(unknown.text().text() == "odd", "unknown label text");
}

void continuation_and_merge() {
  const auto blocks = parse_whole("<BLOCK>cross_table[1,2,3,4]<FORK><table></table><MERGE>"
                                  "<BLOCK>Append_Text[1,2,3,4]<FORK>tail");
  require(blocks[0].continuation() == v1::CONTINUATION_CONTINUES_ON_NEXT_PAGE, "cross_");
  require(blocks[1].continuation() == v1::CONTINUATION_CONTINUED_FROM_PREVIOUS, "append_, any case");
  require(blocks[1].kind() == v1::BLOCK_KIND_TEXT && blocks[1].label_text() == "Append_Text", "label as written");
  require(!blocks[0].merges_with_previous() && blocks[1].merges_with_previous(), "<MERGE> joins the next block");
  require(blocks[0].table().source_html() == "<table></table><MERGE>", "source HTML keeps what was written");
}

void kinds_and_content() {
  const auto eq = only(parse_whole("<BLOCK>equation[1,2,3,4]<FORK>$$x^2$$"));
  require(eq.has_formula() && eq.formula().latex() == "$$x^2$$", "equation");
  const auto ref = only(parse_whole("<BLOCK>ref_text[1,2,3,4]<FORK>[1] A."));
  require(ref.kind() == v1::BLOCK_KIND_REFERENCE, "ref_text");
  const auto seal = only(parse_whole("<BLOCK>seal[1,2,3,4]<FORK>APPROVED"));
  require(seal.kind() == v1::BLOCK_KIND_SEAL && seal.picture().text() == "APPROVED", "seal keeps its text");
  require(output::block_kind("aside_text") == v1::BLOCK_KIND_ASIDE, "aside");
  require(output::block_kind("algorithm") == v1::BLOCK_KIND_ALGORITHM, "algorithm");
  require(output::block_kind("page_number") == v1::BLOCK_KIND_PAGE_NUMBER, "page number");
  const auto stray = only(parse_whole("<BLOCK>text[1,2,3,4]<FORK>a<image>b<|ref|>c"));
  require(stray.text().text() == "abc", "stray control tokens are removed from text");
}

void steps_point_back() {
  const std::vector<std::string> pieces = {"<BLOCK>", "text", "[1,2,3,4]", "<FORK>", "hello", "\n", "<BLOCK>",
                                           "title", "[5,6,7,8]", "<FORK>", "world"};
  const auto blocks = parse_pieces(pieces);
  require(blocks.size() == 2, "two blocks");
  require(blocks[0].first_step() == 0 && blocks[0].last_step() == 5, "first block steps 0..5");
  require(blocks[1].first_step() == 6 && blocks[1].last_step() == 10, "second block steps 6..10");
}

}  // namespace

int main() {
  return run_tests([] {
    golden();
    level_and_box_order();
    raw_boxes_kept();
    bare_labels();
    unrecognised_text();
    continuation_and_merge();
    kinds_and_content();
    steps_point_back();
  });
}
