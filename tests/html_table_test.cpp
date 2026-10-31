// The HTML table parser: rows, header cells, spans placed as a browser
// places them, and upstream's span repair.

#include "check.h"
#include "output/html_table.h"

using namespace uocr::output;

namespace {

void spans_and_headers() {
  const auto t = parse_html_table(
      "<table><thead><tr><th colspan=\"2\">Name</th><th>Qty</th></tr></thead>"
      "<tbody><tr><td rowspan=\"2\">A</td><td>x</td><td>1</td></tr>"
      "<tr><td>y</td><td>2</td></tr></tbody></table>");
  require(t.row_count() == 3 && t.column_count() == 3, "3 x 3 grid");
  require(t.cells_size() == 7, "seven source cells");
  require(t.cells(0).header() && t.cells(0).col_span() == 2 && t.cells(1).column() == 2, "header row spans");
  require(t.cells(2).text() == "A" && t.cells(2).row_span() == 2 && !t.cells(2).header(), "row span");
  const auto& y = t.cells(5);
  require(y.text() == "y" && y.row() == 2 && y.column() == 1, "a cell placed past the row span");
}

void repairs_and_text() {
  const auto t = parse_html_table(
      "<table><tr><td colspan=\"2a\">a &amp; b<br>c</td><td rowspan=\"x\">&#x41;&#66;</td>"
      "<TD COLSPAN='3'>loud</TD></tr></table>");
  require(t.cells(0).col_span() == 2, "colspan=\"2a\" reads as 2");
  require(t.cells(0).text() == "a & b\nc", "entities decoded, <br> is a newline");
  require(t.cells(1).row_span() == 1 && t.cells(1).text() == "AB", "rowspan=\"x\" reads as 1");
  require(t.cells(2).col_span() == 3 && t.cells(2).column() == 3, "tags in any case, single quotes");
  require(t.column_count() == 6, "column count");
}

void malformed() {
  require(parse_html_table("").cells_size() == 0, "empty");
  require(parse_html_table("<table><tr><td>open").row_count() == 0, "an unclosed row is not read");
  const auto t = parse_html_table("<tr><td>a<td>b</tr><track><tr><th>h</tr>");
  require(t.row_count() == 2 && t.cells_size() == 3, "unclosed cells end at the next cell");
  require(t.cells(1).text() == "b" && t.cells(2).header(), "cells");
  const auto big = parse_html_table("<tr><td colspan=\"999999999\">x</td></tr>");
  require(big.cells(0).col_span() == 1000, "spans are clamped");
}

}  // namespace

int main() {
  return run_tests([] {
    spans_and_headers();
    repairs_and_text();
    malformed();
  });
}
