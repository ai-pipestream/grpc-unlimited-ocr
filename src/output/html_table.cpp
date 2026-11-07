#include "output/html_table.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace uocr::output {
namespace {

namespace v1 = ::ai::pipestream::unlimitedocr::v1;

constexpr std::size_t npos = std::string_view::npos;
// Spans past this are clamped: a looping model can write colspan="99999",
// and placement marks every spanned column.
constexpr std::uint32_t kMaxSpan = 1000;

char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }
bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

bool at_ci(std::string_view s, std::size_t p, std::string_view word) {
  if (p > s.size() || s.size() - p < word.size()) return false;
  for (std::size_t i = 0; i < word.size(); ++i) {
    if (lower(s[p + i]) != word[i]) return false;
  }
  return true;
}

std::size_t find_ci(std::string_view s, std::string_view word, std::size_t from) {
  for (std::size_t p = s.find('<', from); p != npos; p = s.find('<', p + 1)) {
    if (at_ci(s, p, word)) return p;
  }
  return npos;
}

// "<name" followed by whitespace, '>' or '/', so <tr is not <track.
bool opens(std::string_view s, std::size_t p, std::string_view open_name) {
  if (!at_ci(s, p, open_name)) return false;
  const std::size_t q = p + open_name.size();
  return q < s.size() && (is_space(s[q]) || s[q] == '>' || s[q] == '/');
}

std::size_t find_open(std::string_view s, std::string_view open_name, std::size_t from) {
  for (std::size_t p = find_ci(s, open_name, from); p != npos; p = find_ci(s, open_name, p + 1)) {
    if (opens(s, p, open_name)) return p;
  }
  return npos;
}

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp == 0 || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// An entity at `p` ('&'): appends its text and returns its length, or 0.
std::size_t entity(std::string_view s, std::size_t p, std::string& out) {
  const std::size_t semi = s.find(';', p);
  if (semi == npos || semi - p > 10) return 0;
  const std::string_view name = s.substr(p + 1, semi - p - 1);
  static constexpr std::pair<std::string_view, std::string_view> kNamed[] = {
      {"amp", "&"}, {"lt", "<"}, {"gt", ">"}, {"quot", "\""}, {"apos", "'"}, {"nbsp", "\xC2\xA0"}};
  for (const auto& [n, text] : kNamed) {
    if (name == n) {
      out.append(text);
      return semi - p + 1;
    }
  }
  if (name.size() < 2 || name[0] != '#') return 0;
  const bool hex = name[1] == 'x' || name[1] == 'X';
  const std::string_view digits = name.substr(hex ? 2 : 1);
  if (digits.empty()) return 0;
  std::uint32_t cp = 0;
  for (const char c : digits) {
    const int d = c >= '0' && c <= '9' ? c - '0' : hex && lower(c) >= 'a' && lower(c) <= 'f' ? lower(c) - 'a' + 10 : -1;
    if (d < 0) return 0;
    cp = std::min<std::uint32_t>(cp * (hex ? 16 : 10) + static_cast<std::uint32_t>(d), 0x110000);
  }
  append_utf8(out, cp);
  return semi - p + 1;
}

// Cell text: tags removed (<br> becomes a newline), entities decoded,
// outer whitespace trimmed.
std::string cell_text(std::string_view s) {
  std::string out;
  for (std::size_t p = 0; p < s.size();) {
    if (s[p] == '<') {
      const std::size_t close = s.find('>', p);
      if (close == npos) break;
      if (opens(s, p, "<br")) out.push_back('\n');
      p = close + 1;
    } else if (s[p] == '&') {
      const std::size_t n = entity(s, p, out);
      if (n == 0) out.push_back('&');
      p += n == 0 ? 1 : n;
    } else {
      out.push_back(s[p++]);
    }
  }
  const std::size_t b = out.find_first_not_of(" \t\r\n\f\v");
  if (b == std::string::npos) return {};
  return out.substr(b, out.find_last_not_of(" \t\r\n\f\v") - b + 1);
}

// colspan / rowspan from an opening tag, repaired as upstream does: a value
// that is not a number becomes its first run of digits, or 1.
std::uint32_t span_attribute(std::string_view open_tag, std::string_view name) {
  std::size_t p = 0;
  while (true) {
    std::size_t at = npos;
    for (std::size_t q = p; q + name.size() <= open_tag.size(); ++q) {
      if (at_ci(open_tag, q, name) && (q == 0 || !std::isalnum(static_cast<unsigned char>(open_tag[q - 1])))) {
        at = q;
        break;
      }
    }
    if (at == npos) return 1;
    std::size_t q = at + name.size();
    while (q < open_tag.size() && is_space(open_tag[q])) ++q;
    if (q >= open_tag.size() || open_tag[q] != '=') {
      p = at + 1;
      continue;
    }
    ++q;
    while (q < open_tag.size() && is_space(open_tag[q])) ++q;
    const char quote = q < open_tag.size() && (open_tag[q] == '"' || open_tag[q] == '\'') ? open_tag[q] : '\0';
    if (quote != '\0') ++q;
    std::size_t end = q;
    while (end < open_tag.size() && (quote != '\0' ? open_tag[end] != quote : !is_space(open_tag[end]) && open_tag[end] != '>')) ++end;
    const std::string_view value = open_tag.substr(q, end - q);
    const std::size_t d = value.find_first_of("0123456789");
    if (d == npos) return 1;
    std::uint32_t v = 0;
    for (std::size_t i = d; i < value.size() && value[i] >= '0' && value[i] <= '9'; ++i) {
      v = std::min<std::uint32_t>(v * 10 + static_cast<std::uint32_t>(value[i] - '0'), kMaxSpan);
    }
    return std::clamp<std::uint32_t>(v, 1, kMaxSpan);
  }
}

struct Cell {
  bool header = false;
  std::uint32_t row_span = 1;
  std::uint32_t col_span = 1;
  std::string text;
};

// The cells of one row's content, in order.
std::vector<Cell> row_cells(std::string_view row) {
  std::vector<Cell> cells;
  std::size_t p = 0;
  while (true) {
    const std::size_t td = find_open(row, "<td", p);
    const std::size_t th = find_open(row, "<th", p);
    const std::size_t at = std::min(td, th);
    if (at == npos) break;
    const bool header = at == th;
    const std::size_t open_end = row.find('>', at);
    if (open_end == npos) break;
    const std::size_t next_td = find_open(row, "<td", open_end);
    const std::size_t next_th = find_open(row, "<th", open_end);
    const std::size_t close = find_ci(row, header ? "</th" : "</td", open_end);
    const std::size_t end = std::min({close, next_td, next_th, row.size()});
    const std::string_view open_tag = row.substr(at, open_end - at);
    cells.push_back({header, span_attribute(open_tag, "rowspan"), span_attribute(open_tag, "colspan"),
                     cell_text(row.substr(open_end + 1, end - open_end - 1))});
    p = end;
  }
  return cells;
}

}  // namespace

v1::TableContent parse_html_table(std::string_view html) {
  v1::TableContent table;
  std::vector<std::uint32_t> busy_until;  // per column: the first row it is free again
  std::uint32_t row = 0;
  std::uint32_t columns = 0;
  // The <thead> section ahead of (or around) the current row.
  std::size_t head_open = find_open(html, "<thead", 0);
  std::size_t head_close = head_open == npos ? npos : find_ci(html, "</thead", head_open);
  for (std::size_t p = find_open(html, "<tr", 0); p != npos;) {
    const std::size_t open_end = html.find('>', p);
    if (open_end == npos) break;
    const std::size_t close = find_ci(html, "</tr", open_end);
    if (close == npos) break;  // upstream reads only closed rows
    while (head_close != npos && head_close < p) {
      head_open = find_open(html, "<thead", head_close);
      head_close = head_open == npos ? npos : find_ci(html, "</thead", head_open);
    }
    const bool in_head = head_open != npos && head_open < p;
    std::uint32_t col = 0;
    for (const Cell& cell : row_cells(html.substr(open_end + 1, close - open_end - 1))) {
      while (col < busy_until.size() && busy_until[col] > row) ++col;
      if (busy_until.size() < col + cell.col_span) busy_until.resize(col + cell.col_span, 0);
      std::fill_n(busy_until.begin() + col, cell.col_span, row + cell.row_span);
      v1::TableCellContent* out = table.add_cells();
      out->set_row(row);
      out->set_column(col);
      out->set_row_span(cell.row_span);
      out->set_col_span(cell.col_span);
      out->set_header(cell.header || in_head);
      out->set_text(cell.text);
      col += cell.col_span;
      columns = std::max(columns, col);
    }
    ++row;
    p = find_open(html, "<tr", close);
  }
  table.set_row_count(row);
  table.set_column_count(columns);
  return table;
}

}  // namespace uocr::output
