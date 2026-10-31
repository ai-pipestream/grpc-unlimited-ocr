// The Document fold: page fragments keep every block as its own item; the
// folded document joins what the continuation links join (a table across
// pages with its repeated header dropped, a continued paragraph, merged
// columns), with provenance on every page.

#include "check.h"
#include "output/continuations.h"
#include "output/document_fold.h"
#include "output_helpers.h"

using namespace uocr;
using namespace uocr::testing;

namespace {

namespace docv1 = ::ai::pipestream::document::v1;

struct Parsed {
  std::vector<v1::Block> blocks;
  std::vector<v1::ContinuationLink> links;
};

Parsed parse_pages(const std::vector<std::string>& pages) {
  Parsed out;
  output::ContinuationTracker tracker;
  for (std::size_t i = 0; i < pages.size(); ++i) {
    for (auto& b : parse_text(pages[i])) {
      b.set_page_no(static_cast<std::uint32_t>(i + 1));
      for (auto& l : tracker.add(b)) out.links.push_back(l);
      out.blocks.push_back(b);
    }
  }
  return out;
}

output::DocumentInput input(std::uint32_t pages) {
  output::DocumentInput in;
  in.name = "doc";
  in.version = "test";
  for (std::uint32_t p = 1; p <= pages; ++p) in.pages.push_back({p, 1000, 2000});
  in.generation.prompt_tokens = 10;
  in.generation.completion_tokens = 20;
  return in;
}

void cross_page_table() {
  const auto parsed = parse_pages({
      "<BLOCK>cross_table[0,0,1000,500]<FORK><table><tr><th>H</th><th>V</th></tr><tr><td>a</td><td>1</td></tr>"
      "</table>",
      "<BLOCK>append_table[0,0,1000,500]<FORK><table><tr><th>H</th><th>V</th></tr><tr><td>b</td><td>2</td></tr>"
      "</table>",
  });
  require(parsed.links.size() == 1, "one cross-page link");
  require(parsed.links[0].link_kind() == v1::CONTINUATION_LINK_KIND_CROSS_PAGE, "link kind");
  require(parsed.links[0].from_page_no() == 1 && parsed.links[0].to_page_no() == 2, "page 1 to page 2");
  const docv1::Document doc = output::fold_document(input(2), parsed.blocks, parsed.links);
  require(doc.tables_size() == 1, "one folded table");
  const auto& table = doc.tables(0);
  require(table.prov_size() == 2 && table.prov(1).page_no() == 2, "provenance on both pages");
  require(table.prov(0).bbox().b() == 1000.0, "boxes scale to page pixels");
  require(table.data().num_rows() == 3 && table.data().num_cols() == 2, "the repeated header row is dropped");
  require(table.data().table_cells(4).text() == "b" && table.data().table_cells(4).start_row_offset_idx() == 2,
          "the second fragment's rows follow");
  require(table.data().grid_size() == 3, "a full grid");
  const docv1::Document page = output::page_document(input(1), {parsed.blocks[0]});
  require(page.tables_size() == 1 && page.tables(0).data().num_rows() == 2, "a fragment stays page-local");
}

void continued_text_and_merge() {
  const auto parsed = parse_pages({
      "<BLOCK>title<LEVEL2>[0,0,10,10]<FORK>Section\n<BLOCK>cross_text[0,0,10,10]<FORK>The para",
      "<BLOCK>append_text[0,0,10,10]<FORK>graph ends.<MERGE><BLOCK>text[0,0,10,10]<FORK>Column two.",
  });
  require(parsed.links.size() == 2, "a cross-page link and a merge link");
  require(parsed.links[1].link_kind() == v1::CONTINUATION_LINK_KIND_MERGE, "merge");
  const docv1::Document doc = output::fold_document(input(2), parsed.blocks, parsed.links);
  require(doc.texts_size() == 2, "the section header and one joined paragraph");
  require(doc.texts(0).has_section_header() && doc.texts(0).section_header().level() == 2, "level 2 header");
  const auto& para = doc.texts(1).text().base();
  require(para.text() == "The paragraph ends.Column two.", "joined as upstream joins: no separator");
  require(para.prov_size() == 3, "provenance per block");
  require(doc.body().children_size() == 2 && doc.body().children(1).ref() == "#/texts/1", "body order");
}

void labels_and_sources() {
  output::DocumentInput in = input(1);
  const std::vector<float> logprobs(64, -0.25f);
  in.step_logprobs = logprobs;
  const auto parsed = parse_pages({
      "<BLOCK>header[0,0,1,1]<FORK>Running head\n<BLOCK>image_caption[0,0,1,1]<FORK>Fig 1\n"
      "<BLOCK>figure[0,0,1,1]<FORK>\n<BLOCK>algorithm[0,0,1,1]<FORK>x = 1\n<BLOCK>title[0,0,1,1]<FORK>Top\n"
      "<BLOCK>page_number[0,0,1,1]<FORK>7\n<BLOCK>weird[0,0,1,1]<FORK>?",
  });
  const docv1::Document doc = output::page_document(in, parsed.blocks);
  require(doc.texts(0).text().base().label() == docv1::DOC_ITEM_LABEL_PAGE_HEADER, "header");
  require(doc.texts(0).text().base().content_layer() == docv1::CONTENT_LAYER_FURNITURE, "furniture");
  require(doc.texts(1).text().base().label() == docv1::DOC_ITEM_LABEL_CAPTION, "caption");
  require(doc.pictures_size() == 1 && doc.pictures(0).captions_size() == 1, "a caption before its picture");
  require(doc.pictures(0).captions(0).ref() == "#/texts/1", "caption ref");
  require(doc.texts(2).has_code(), "algorithm as code");
  require(doc.texts(3).has_title(), "a title without level is the title");
  require(doc.texts(4).text().base().label() == docv1::DOC_ITEM_LABEL_PAGE_FOOTER, "page number is footer");
  require(doc.texts(5).text().base().label_raw() == "weird", "unknown label kept raw");
  const auto& source = doc.texts(0).text().base().source();
  require(source.size() == 2 && source[0].collector().collector() == "unlimited-ocr", "collector source");
  require(source[0].collector().raw_score() == -0.25 && source[0].collector().raw_score_kind() == "mean_logprob",
          "the model's mean log-probability");
  require(source[1].generation().model() == "Unlimited-OCR-2" && source[1].generation().prompt_tokens() == 10,
          "generation source");
  require(doc.pages().at(1).size().height() == 2000, "page size");
}

}  // namespace

int main() {
  return run_tests([] {
    cross_page_table();
    continued_text_and_merge();
    labels_and_sources();
  });
}
