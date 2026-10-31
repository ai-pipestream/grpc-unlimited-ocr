#include "output/block_grammar.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>

namespace uocr::output {
namespace {

// Bytes 0..end that no block header claims, kept verbatim.
BlockSpan unrecognised_span(std::size_t end) {
  BlockSpan span;
  span.end = end;
  return span;
}

// The text is scanned by hand rather than with std::regex: libstdc++'s
// engine recurses per character and a looping model's block can be tens
// of kilobytes. Each helper mirrors one piece of upstream's patterns and
// says which; `npos` means no match.
constexpr std::size_t npos = std::string_view::npos;

constexpr std::string_view kFullWidthComma = "\xEF\xBC\x8C";  // ，
constexpr std::string_view kFullWidthBar = "\xEF\xBD\x9C";    // ｜

constexpr std::array<std::string_view, 17> kBaseLabels = {
    "text",  "title", "header", "footer",        "page_number",    "image",         "figure",   "chart",    "seal",
    "table", "image_caption",   "image_footnote", "page_footnote", "equation",      "ref_text", "aside_text",
    "algorithm"};

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alpha_(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

std::size_t skip_space(std::string_view s, std::size_t p) {
  while (p < s.size() && is_space(s[p])) ++p;
  return p;
}

std::size_t skip_digits(std::string_view s, std::size_t p) {
  while (p < s.size() && is_digit(s[p])) ++p;
  return p;
}

// `word` (lowercase ASCII) at `p`, ignoring case.
bool at_word(std::string_view s, std::size_t p, std::string_view word) {
  if (p > s.size() || s.size() - p < word.size()) return false;
  for (std::size_t i = 0; i < word.size(); ++i) {
    if (lower(s[p + i]) != word[i]) return false;
  }
  return true;
}

std::string lowered(std::string_view s) {
  std::string out(s);
  std::ranges::transform(out, out.begin(), lower);
  return out;
}

// A ',' or a full-width '，' at `p`: its length, or 0.
std::size_t comma(std::string_view s, std::size_t p) {
  if (p < s.size() && s[p] == ',') return 1;
  return s.substr(p).starts_with(kFullWidthComma) ? kFullWidthComma.size() : 0;
}

// A '|' or a full-width '｜' at `p`: its length, or 0.
std::size_t bar(std::string_view s, std::size_t p) {
  if (p < s.size() && s[p] == '|') return 1;
  return s.substr(p).starts_with(kFullWidthBar) ? kFullWidthBar.size() : 0;
}

enum class Closer { kAngle, kOptionalAngleOrBrace };

// <\s*/?\s*NAME\s*(\d*\s*)?CLOSER, where digits are allowed only when
// `digits` is set and a slash only when `slash` is. Returns the end and,
// through `number`, the digits' span.
std::size_t tag(std::string_view s, std::size_t p, std::string_view name, bool slash, bool digits, Closer closer,
                std::string_view* number = nullptr) {
  if (p >= s.size() || s[p] != '<') return npos;
  std::size_t q = skip_space(s, p + 1);
  if (slash && q < s.size() && s[q] == '/') q = skip_space(s, q + 1);
  if (!at_word(s, q, name)) return npos;
  q = skip_space(s, q + name.size());
  if (digits) {
    const std::size_t d = skip_digits(s, q);
    if (number != nullptr) *number = s.substr(q, d - q);
    q = skip_space(s, d);
  }
  if (q < s.size() && (s[q] == '>' || (closer == Closer::kOptionalAngleOrBrace && s[q] == '}'))) return q + 1;
  return closer == Closer::kAngle ? npos : q;
}

// <LEVELn> as upstream reads it: <\s*/?\s*LEVEL\s*(\d*)\s*[>}]?
std::size_t level_tag(std::string_view s, std::size_t p, std::string_view* number = nullptr) {
  return tag(s, p, "level", true, true, Closer::kOptionalAngleOrBrace, number);
}

// -?\d+ at `p`, saturated to int32.
std::size_t integer(std::string_view s, std::size_t p, std::int32_t* value) {
  std::size_t q = p;
  const bool negative = q < s.size() && s[q] == '-';
  if (negative) ++q;
  const std::size_t end = skip_digits(s, q);
  if (end == q) return npos;
  std::int64_t v = 0;
  for (std::size_t i = q; i < end; ++i) {
    v = std::min<std::int64_t>(v * 10 + (s[i] - '0'), std::int64_t{std::numeric_limits<std::int32_t>::max()} + 1);
  }
  if (negative) v = -v;
  *value = static_cast<std::int32_t>(std::clamp<std::int64_t>(v, std::numeric_limits<std::int32_t>::min(),
                                                               std::numeric_limits<std::int32_t>::max()));
  return end;
}

// \[\s*-?\d+(\s*[,，]\s*-?\d+){3,} : the opening bracket and at least four
// integers. Returns the end of the last integer.
std::size_t box_numbers(std::string_view s, std::size_t p, std::vector<std::int32_t>* values) {
  if (p >= s.size() || s[p] != '[') return npos;
  std::int32_t v = 0;
  std::size_t q = integer(s, skip_space(s, p + 1), &v);
  if (q == npos) return npos;
  values->assign(1, v);
  while (true) {
    const std::size_t c = skip_space(s, q);
    const std::size_t width = comma(s, c);
    if (width == 0) break;
    const std::size_t next = integer(s, skip_space(s, c + width), &v);
    if (next == npos) break;
    values->push_back(v);
    q = next;
  }
  return values->size() >= 4 ? q : npos;
}

// The box inside a <BLOCK> header: the numbers, then \s*\] .
std::size_t strict_box(std::string_view s, std::size_t p, std::vector<std::int32_t>* values) {
  const std::size_t q = box_numbers(s, p, values);
  if (q == npos) return npos;
  const std::size_t close = skip_space(s, q);
  return close < s.size() && s[close] == ']' ? close + 1 : npos;
}

// The box of a bare declaration: the numbers, then [^\]]*\] . Unlike
// upstream the trailing junk may not cross a newline or a '<': that keeps
// a pending declaration from reaching over a later block's start, which
// is what makes every closed block final.
std::size_t bare_box(std::string_view s, std::size_t p) {
  std::vector<std::int32_t> ignored;
  std::size_t q = box_numbers(s, p, &ignored);
  if (q == npos) return npos;
  while (q < s.size() && s[q] != ']' && s[q] != '\n' && s[q] != '<') ++q;
  return q < s.size() && s[q] == ']' ? q + 1 : npos;
}

// Length of a layout control token at `p` (upstream's STRAY_TOKEN_RE,
// alternatives in its order), or 0.
std::size_t stray_token(std::string_view s, std::size_t p) {
  if (p >= s.size() || s[p] != '<') return 0;
  if (const std::size_t b = bar(s, p + 1); b != 0) {
    std::size_t q = p + 1 + b;
    if (q < s.size() && s[q] == '/') ++q;
    const std::size_t name = q;
    while (q < s.size() && is_alpha_(s[q])) ++q;
    if (q > name) {
      if (const std::size_t e = bar(s, q); e != 0 && q + e < s.size() && s[q + e] == '>') return q + e + 1 - p;
    }
  }
  if (at_word(s, p, "<image>")) return 7;
  for (const auto& [name, digits] : {std::pair{"page", true}, {"block", false}, {"fork", false}}) {
    if (const std::size_t e = tag(s, p, name, true, digits, Closer::kAngle); e != npos) return e - p;
  }
  if (const std::size_t e = level_tag(s, p); e != npos) return e - p;
  if (const std::size_t e = tag(s, p, "merge", true, false, Closer::kAngle); e != npos) return e - p;
  return 0;
}

bool contains_merge(std::string_view s) {
  for (std::size_t p = s.find('<'); p != npos; p = s.find('<', p + 1)) {
    if (tag(s, p, "merge", true, false, Closer::kAngle) != npos) return true;
  }
  return false;
}

bool known_label(std::string_view label) {
  std::string base = lowered(label);
  for (const std::string_view prefix : {"cross_", "append_"}) {
    if (base.starts_with(prefix)) {
      base.erase(0, prefix.size());
      break;
    }
  }
  return std::ranges::find(kBaseLabels, base) != kBaseLabels.end();
}

// A block header from its label onwards: label, level and box in either
// order, then an optional <FORK>. Fills `span` and returns the header end.
std::size_t header_from_label(std::string_view s, std::size_t p, BlockSpan* span) {
  std::size_t q = p;
  while (q < s.size() && is_alpha_(s[q])) ++q;
  if (q == p) return npos;
  span->label = std::string(s.substr(p, q - p));
  std::string_view digits;
  auto take_level = [&](std::size_t end) {
    span->has_level = true;
    if (!digits.empty()) {
      std::int32_t v = 0;
      integer(digits, 0, &v);
      span->level = v;
    }
    return end;
  };
  if (const std::size_t l = level_tag(s, skip_space(s, q), &digits); l != npos) {
    q = take_level(l);
    if (const std::size_t b = strict_box(s, skip_space(s, q), &span->raw_box); b != npos) q = b;
  } else if (const std::size_t b = strict_box(s, skip_space(s, q), &span->raw_box); b != npos) {
    q = b;
    if (const std::size_t l2 = level_tag(s, skip_space(s, q), &digits); l2 != npos) q = take_level(l2);
  }
  q = skip_space(s, q);
  if (const std::size_t f = tag(s, q, "fork", false, false, Closer::kAngle); f != npos) q = f;
  return q;
}

struct Start {
  std::size_t at = 0;     // where the block begins
  std::size_t label = 0;  // where its label begins
};

// Bare declarations: "label[x1,y1,x2,y2]" at the page start, after a
// newline or after <FORK>, optionally behind <|det|>, with a known label.
std::vector<Start> bare_starts(std::string_view s) {
  std::vector<Start> out;
  for (std::size_t p = 0; p < s.size();) {
    const bool anchored = p == 0 || s[p - 1] == '\n' || (p >= 6 && at_word(s, p - 6, "<fork>"));
    if (!anchored) {
      ++p;
      continue;
    }
    std::size_t q = p;
    if (const std::size_t b = bar(s, q + 1); q < s.size() && s[q] == '<' && b != 0) {
      std::size_t d = skip_space(s, q + 1 + b);
      if (at_word(s, d, "det")) {
        d = skip_space(s, d + 3);
        if (const std::size_t e = bar(s, d); e != 0 && d + e < s.size() && s[d + e] == '>') q = d + e + 1;
      }
    }
    const std::size_t label = q;
    if (q < s.size() && is_alpha_(s[q])) {
      while (q < s.size() && (is_alpha_(s[q]) || is_digit(s[q]))) ++q;
    }
    std::size_t end = npos;
    if (q > label) {
      std::size_t r = q;
      if (const std::size_t l = level_tag(s, skip_space(s, r)); l != npos) r = l;
      end = bare_box(s, skip_space(s, r));
    }
    if (end == npos) {
      ++p;
      continue;
    }
    if (known_label(s.substr(label, q - label))) out.push_back({label, label});
    p = end;
  }
  return out;
}

// <BLOCK> tags followed by a label.
std::vector<Start> tagged_starts(std::string_view s) {
  std::vector<Start> out;
  for (std::size_t p = s.find('<'); p != npos; p = s.find('<', p + 1)) {
    const std::size_t t = tag(s, p, "block", false, false, Closer::kAngle);
    if (t == npos) continue;
    const std::size_t label = skip_space(s, t);
    if (label < s.size() && is_alpha_(s[label])) out.push_back({p, label});
  }
  return out;
}

}  // namespace

std::string_view trim(std::string_view text) {
  std::size_t b = 0;
  std::size_t e = text.size();
  while (b < e && is_space(text[b])) ++b;
  while (e > b && is_space(text[e - 1])) --e;
  return text.substr(b, e - b);
}

std::string clean_content(std::string_view content) {
  std::string out;
  out.reserve(content.size());
  for (std::size_t p = 0; p < content.size();) {
    if (const std::size_t n = stray_token(content, p); n != 0) {
      p += n;
      continue;
    }
    out.push_back(content[p++]);
  }
  return std::string(trim(out));
}

std::vector<BlockSpan> split_blocks(std::string_view page) {
  std::vector<Start> starts = tagged_starts(page);
  std::ranges::move(bare_starts(page), std::back_inserter(starts));
  std::ranges::sort(starts, {}, &Start::at);

  std::vector<BlockSpan> spans;
  for (const Start& start : starts) {
    if (!spans.empty() && start.at < spans.back().content_begin) continue;  // inside the previous header
    BlockSpan span;
    span.begin = start.at;
    span.recognised = true;
    span.content_begin = header_from_label(page, start.label, &span);
    if (span.content_begin == npos) continue;
    spans.push_back(std::move(span));
  }
  if (spans.empty()) {
    if (!page.empty()) spans.push_back(unrecognised_span(page.size()));
    return spans;
  }
  for (std::size_t j = 0; j < spans.size(); ++j) {
    spans[j].end = j + 1 < spans.size() ? spans[j + 1].begin : page.size();
    spans[j].content_begin = std::min(spans[j].content_begin, spans[j].end);
    const std::size_t gap_begin = j == 0 ? 0 : spans[j - 1].content_begin;
    spans[j].merge_with_previous = contains_merge(page.substr(gap_begin, spans[j].begin - gap_begin));
  }
  // Text before the first block: its own span when anything is left once
  // the control tokens are gone, otherwise part of the first block so
  // every byte stays covered.
  if (const std::string_view pre = page.substr(0, spans.front().begin); !pre.empty()) {
    if (clean_content(pre).empty()) {
      spans.front().begin = 0;
    } else {
      spans.insert(spans.begin(), unrecognised_span(pre.size()));
    }
  }
  return spans;
}

}  // namespace uocr::output
