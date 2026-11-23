"""Turn the handbook PDFs in Data/ into chunks.json for search.

There are two kinds of PDF in Data/:

* Typeset documents (the IOM manual, the safety register, WP-114, OV-220).
  These are read with pdfplumber, so tables come out row by row, the running
  header and footer on every page are dropped, and lines are joined back into
  paragraphs.

* The parts catalogue, a fixed-width printout from a parts database (set in
  Courier). It is read line by line with pypdf. The **** banner, the page
  header repeated above every table, the page footers and the printer control
  codes are removed, and each part row stays on one line, so a part number is
  never separated from its description.

Both are turned into a list of "units": a heading, a paragraph, a table row or
a catalogue line. Chunks are built from whole units, so a chunk never ends in
the middle of a word or a row. When a chunk starts partway through a section,
it begins with a context line in [brackets] (the section heading, or the
catalogue model and variant) and, inside a table, the table's header row.

The catalogue changes pump model partway through some pages (page 41 ends
the HX-800 tables, then starts HX-250 and then HX-400). Every catalogue unit
carries the product group from the most recent "PRODUCT GROUP:" line, every
catalogue chunk starts with that group, e.g. "[HX-250]", and a chunk never holds
two groups: when the group changes, a new chunk starts. A page header without a
"PRODUCT GROUP:" line opens a section that covers every model (stock holding,
part-number index, torque table, ...); those chunks start with "[ALL MODELS]".
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "Data"
OUTPUT_FILE = BASE_DIR / "backend" / "chunks.json"

CHUNK_SIZE = 1000  # max characters per chunk
MIN_CHUNK = 300    # a new heading starts a new chunk once the current one is this long

# Layout of the typeset PDFs (A4, in points).
HEADER_MARGIN = 50   # running header sits above this line
FOOTER_MARGIN = 45   # running footer sits in the bottom 45 pt
PARAGRAPH_GAP = 8    # a bigger vertical gap between two lines starts a new paragraph
HEADING_SIZE = 11.5  # bold text at least this size is a heading
BULLET = "(cid:127)"  # how pdfplumber reads the bullet character in these PDFs
CAPTION = re.compile(r"(Table|Tabel) \d")  # a table caption, e.g. "Table 5.4 - ..."
NOTE_PREFIXES = ("NOTE", "WARNING")  # catalogue lines that are notes, not table rows


@dataclass
class Unit:
    text: str
    page: int
    kind: str = "text"  # "heading", "text" or "row"; the catalogue also uses "note" and "para"
    section: str = ""
    context: list = field(default_factory=list)  # lines to put first if a chunk starts here
    product_group: str = ""  # catalogue only, e.g. "HX-250"; a chunk never mixes two


def bracket(section):
    return [f"[{section}]"] if section else []


# ---------------------------------------------------------------- typeset PDFs

def clean(text):
    text = text.replace(BULLET, "•")
    # A table cell too narrow for a word pushes its last letter onto the next
    # line ("Bedrijfsvor" / "m"). Glue a lone lowercase letter back on.
    text = re.sub(r"(?<=[A-Za-z])\n([a-z])(?=\n|$)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def is_heading(line):
    chars = [c for c in line["chars"] if c["text"].strip()]
    return all("Bold" in c["fontname"] for c in chars) and chars[0]["size"] >= HEADING_SIZE


def inside(line, bbox):
    middle = (line["top"] + line["bottom"]) / 2
    return bbox[1] <= middle <= bbox[3] and line["x0"] < bbox[2] and line["x1"] > bbox[0]


def split_long(text, limit=CHUNK_SIZE - 200):
    """Split a very long paragraph into pieces at sentence ends."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•(])", text):
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return pieces + [current] if current else pieces


def read_typeset_pdf(path):
    units = []
    headings = []      # current heading path, e.g. ["5 Installation", "5.2 Foundation"]
    paragraph = []     # lines of the paragraph being collected
    paragraph_page = None
    label = ""         # a short all-caps label such as DANGER, joined to the next paragraph
    open_table = None  # a table that ended the previous page and may continue on this one
    last_table = None  # the table just before the current paragraph, to attach its caption

    def section():
        return " > ".join(headings)

    def flush_paragraph():
        nonlocal paragraph, label, last_table
        if not paragraph:
            return
        text = " ".join(paragraph)
        paragraph = []
        if re.fullmatch(r"[A-Z][A-Z ]+", text) and len(text.split()) <= 5:
            label = text  # e.g. "DANGER", "LET OP": put it in front of the next paragraph
            return
        if label:
            text, label = f"{label}: {text}", ""
        if last_table and CAPTION.match(text):
            # Caption under a table: repeat its first sentence when the table is
            # continued in a later chunk.
            caption = re.match(r".*?\.(?=\s|$)", text)
            caption = caption.group(0) if caption else text[:100]
            last_table["header_context"].append(caption)
            last_table["row_context"].insert(-1, caption)
        last_table = None
        for piece in split_long(text):
            units.append(Unit(piece, paragraph_page, "text", section(), bracket(section())))

    def add_table(rows, ncols, page_no, continues):
        """Add a table's rows as units. Returns the table's info for the next page."""
        nonlocal last_table
        cleaned = []
        for row in rows:
            cells = [clean(c) for c in row if c is not None]  # None = part of a merged cell
            while cells and not cells[-1]:
                cells.pop()  # drop empty cells at the end of the row
            if any(cells):
                cleaned.append(" | ".join(cells))
        rows = cleaned
        if not rows:
            return None
        if ncols == 1:  # a boxed note, not a real table
            for row in rows:
                units.append(Unit(row, page_no, "text", section(), bracket(section())))
            return None
        if continues:
            info = open_table
            if rows[0] == info["header"]:
                rows = rows[1:]  # header row repeated at the top of the new page
        else:
            info = {"ncols": ncols, "header": rows[0],
                    "header_context": bracket(section()),
                    "row_context": bracket(section()) + [rows[0]]}
            units.append(Unit(rows[0], page_no, "row", section(), info["header_context"]))
            rows = rows[1:]
        for row in rows:
            units.append(Unit(row, page_no, "row", section(), info["row_context"]))
        last_table = info
        return info

    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            body = page.crop((0, HEADER_MARGIN, page.width, page.height - FOOTER_MARGIN))
            tables = body.find_tables()
            items = [(t.bbox[1], "table", t) for t in tables]
            items += [(line["top"], "line", line) for line in body.extract_text_lines()
                      if not any(inside(line, t.bbox) for t in tables)]
            items.sort(key=lambda item: item[0])

            last_bottom = None
            for position, (_, kind, item) in enumerate(items):
                if kind == "table":
                    flush_paragraph()
                    rows = item.extract()
                    ncols = len(rows[0])
                    continues = position == 0 and open_table is not None and open_table["ncols"] == ncols
                    open_table = add_table(rows, ncols, page_no, continues)
                    last_bottom = item.bbox[3]
                    continue

                open_table = None
                text = clean(item["text"])
                if is_heading(item):
                    flush_paragraph()
                    last_table = None
                    size = item["chars"][0]["size"]
                    if size >= 20:  # document title on the cover page
                        headings = []
                        units.append(Unit(text, page_no, "heading", text))
                    else:
                        parents = [] if size >= 14 else headings[:1]
                        units.append(Unit(text, page_no, "heading", " > ".join(parents + [text]),
                                          bracket(" > ".join(parents))))
                        headings = parents + [text]
                else:
                    if last_bottom is None:
                        # First line on the page: it carries on the paragraph from the
                        # previous page unless that one ended a sentence.
                        new_paragraph = not paragraph or re.search(r"[.:;!?]$", paragraph[-1])
                    else:
                        new_paragraph = item["top"] - last_bottom > PARAGRAPH_GAP
                    if new_paragraph or text.startswith("•"):
                        flush_paragraph()
                    if not paragraph:
                        paragraph_page = page_no
                    paragraph.append(text)
                last_bottom = item["bottom"]
    flush_paragraph()
    return units


# ------------------------------------------------------------ parts catalogue

CATALOGUE_NOISE = re.compile(
    r"^\s*("
    r"=+|-{10,}"                                       # ruled lines
    r"|HALSTROM POMPEN B\.V\.\s{3,}SPARE PARTS.*"      # page header, company line
    r"|HX-SPC-2026\s+EXPORT\s+\S+\s+PAGE\s+\d+"        # page footer
    r"|\*+|\*\s+\*|\*{3} END OF .*\*{3}"               # banner frame, end markers
    r"|\^L|@@PAGEBREAK@@|\.CP \d+|0000FF SPOOL CONTINUES|RPT-HX-SPC/\d+ PAGE EJECT"
    r")\s*$"
)
PRODUCT_GROUP = re.compile(r"PRODUCT GROUP:\s*(HX-\d+)\s+MATERIAL VARIANT:\s*(\S+)")
VARIANT_NAMES = {"/A": "cast iron", "/S": "stainless 316L", "/D": "duplex"}  # from the data sheets
ALL_MODELS = "ALL MODELS"  # product group of a page-header block without a PRODUCT GROUP line

NEW_BLOCK, RULE = "<new block>", "<rule>"


def cells(line):
    """Split a fixed-width line into its columns (two or more spaces apart)."""
    line = re.sub(r"\s{2,}:\s*", ": ", line.strip())  # "DIAMETER      : 250 MM"
    line = line.replace("LEAD CL", "LEAD  CL")        # two header columns printed one space apart
    return re.split(r"\s{2,}", line) if line else []


def has_number(text):
    return any(word[0].isdigit() for word in text.split())


def is_column_header(line, next_line):
    """A header row has two or more columns, no numbers, and numbers in the row below."""
    return (len(cells(line)) >= 2 and not has_number(line)
            and not line.strip().startswith(NOTE_PREFIXES)
            and next_line is not None and has_number(next_line))


def format_row(line, header=None):
    parts = cells(line)
    if header and len(parts) == len(cells(header)) - 1:
        
        for i in reversed(range(len(parts))):
            match = re.fullmatch(r"(.*[A-Za-z].*?)\s+(\d+(?:\.\d+)?)", parts[i])
            if match:
                parts[i:i + 1] = [match.group(1), match.group(2)]
                break
    return " | ".join(parts)


def read_catalogue(path):
    # 1. Every line with its page number. Noise is dropped, but the start of each
    #    page-header block and each ruled line are kept as markers.
    lines = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        for raw in (page.extract_text() or "").split("\n"):
            if CATALOGUE_NOISE.match(raw):
                if raw.startswith("HALSTROM POMPEN B.V."):
                    lines.append((page_no, NEW_BLOCK))
                elif re.fullmatch(r"\s*-{10,}\s*", raw):
                    lines.append((page_no, RULE))
                continue
            banner = re.fullmatch(r"\s*\*\s+(.*?)\s+\*\s*", raw)  # text inside the **** box
            lines.append((page_no, banner.group(1) if banner else raw.rstrip()))

    content = [i for i, (_, text) in enumerate(lines) if text not in (NEW_BLOCK, RULE) and text.strip()]
    next_content = {a: lines[b][1] for a, b in zip(content, content[1:])}

    # 2. Walk the lines, tracking the model, block, sub-table and column header.
    units = []
    model = variant = title = subtitle = header = None
    new_block = False
    rows_in_table = 0
    seen_price_basis = False

    def add(unit):
        # model is the group from the most recent PRODUCT GROUP line. It is reset
        # at every page header, so a header without that line means all models.
        unit.product_group = model or ALL_MODELS
        units.append(unit)

    for i, (page_no, text) in enumerate(lines):
        previous = units[-1] if units else None
        if text == NEW_BLOCK:
            model = variant = title = subtitle = header = None
            new_block = True
            continue
        if text == RULE:
            if header and rows_in_table:
                header = subtitle = None  # a rule after data rows ends the table
            continue
        if not text.strip():
            if previous and previous.kind == "para":
                previous.kind = "text"  # blank line ends a paragraph
            continue
        group = PRODUCT_GROUP.search(text)
        if group:
            model, variant = group.group(1), group.group(2)
            variant = None if variant == "ALL" else variant
            continue
        if text.startswith("PRICE BASIS EUR"):
            if not seen_price_basis:  # repeated in every page header; keep it once
                seen_price_basis = True
                add(Unit(format_row(text), page_no))
            continue

        tag = model or ""
        if variant:
            tag += f" {variant} ({VARIANT_NAMES.get(variant, variant)})"
        column_header = is_column_header(text, next_content.get(i))
        single_cell = len(re.split(r"\s{2,}", text.strip())) == 1

        if new_block:
            # First line under a page header names the block. A parts list starts
            # straight with its column header, so give it a name.
            new_block = False
            if column_header and text.strip().startswith("ITEM"):
                title = "SPARE PARTS LIST"
            else:
                title = format_row(text)
            heading = f"{tag}: {title}" if tag else title
            add(Unit(heading, page_no, "heading", heading))
            if not column_header:
                continue
            previous = units[-1]

        section = f"{tag}: {title}" if tag and title else (title or tag)
        context = bracket(section)

        if column_header:
            if header and cells(text) == cells(header):
                rows_in_table = 0  # so the rule under the repeated header doesn't end the table
                continue           # same header printed again inside the table
            if previous and previous.kind == "text" and not has_number(previous.text):
                subtitle = previous.text  # e.g. "FLANGE BOLTING BY NOMINAL SIZE"
            header, rows_in_table = text, 0
            add(Unit(format_row(text), page_no, "row", section,
                              context + ([subtitle] if subtitle else [])))
            continue

        if text.startswith(("KIT ", "STORE:")):
            # Sub-heading for the rows that follow, e.g. "KIT SEAL KIT | PART NUMBER HX25-KIT-SEAL-GJL"
            subtitle, header = format_row(text), None
            add(Unit(subtitle, page_no, "text", section, context))
            continue
        if subtitle:
            context = context + [subtitle]

        # Prose wrapped over several lines: join a line onto the one before when
        # that one stopped mid-sentence (or it continues an indented NOTE/WARNING).
        if previous and single_cell and len(text.split()) > 3 and (
                (previous.kind == "note" and text.startswith("      "))
                or (previous.kind == "para" and re.search(r"[A-Z0-9,]$", previous.text))):
            previous.text += " " + text.strip()
            continue

        if header and not single_cell:
            rows_in_table += 1
            add(Unit(format_row(text, header), page_no, "row", section,
                              context + [format_row(header)]))
        elif text.strip().startswith(NOTE_PREFIXES):
            add(Unit(format_row(text), page_no, "note", section, context))
        elif single_cell and len(text.split()) >= 5:
            add(Unit(text.strip(), page_no, "para", section, context))
        else:
            add(Unit(format_row(text), page_no, "text", section, context))
    return units


# ------------------------------------------------------------------- chunking

def make_chunks(pdf_path, units):
    chunks, current, per_page = [], [], {}

    def text_of(group):
        body = [u.text for u in group]
        # Skip a repeated table caption when the chunk already ends with that caption.
        context = [c for c in group[0].context
                   if not (CAPTION.match(c) and any(t.startswith(c) for t in body))]
        return "\n".join(bracket(group[0].product_group) + context + body)

    def emit(group):
        first = group[0]
        number = per_page.get(first.page, 0)
        per_page[first.page] = number + 1
        chunks.append({
            "id": f"{pdf_path.stem}-p{first.page}-{number}",
            "document": pdf_path.name,
            "page": first.page,
            "page_end": group[-1].page,
            "section": first.section,
            "text": text_of(group),
        })

    for unit in units:
        if current:
            size = len(text_of(current))
            too_big = size + 1 + len(unit.text) > CHUNK_SIZE
            new_section = unit.kind == "heading" and size >= MIN_CHUNK
            new_group = unit.product_group != current[0].product_group
            if too_big or new_section or new_group:
                carry = []
                while not new_group and current and current[-1].kind == "heading":
                    carry.insert(0, current.pop())  # keep a heading with what follows it
                if current:
                    emit(current)
                current = carry
        current.append(unit)
    if current:
        emit(current)
    return chunks


def is_fixed_width(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        fonts = {c["fontname"] for c in pdf.pages[0].chars}
    return bool(fonts) and all("Courier" in f for f in fonts)


def main():
    chunks = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        units = read_catalogue(pdf_path) if is_fixed_width(pdf_path) else read_typeset_pdf(pdf_path)
        doc_chunks = make_chunks(pdf_path, units)
        print(f"{pdf_path.name}: {len(doc_chunks)} chunks")
        chunks += doc_chunks
    OUTPUT_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
