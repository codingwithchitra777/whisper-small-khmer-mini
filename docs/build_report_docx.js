// Build docs/Final_Report.docx from docs/report.md: title page, contents, headings, lists, tables, figures.
// Usage (once: npm install docx in any folder, then point NODE_PATH at its node_modules):
//   node docs/build_report_docx.js            -> docs/Final_Report.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ExternalHyperlink, HeadingLevel, AlignmentType, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, TableOfContents, Footer, PageNumber,
  LevelFormat, TableLayoutType,
} = require("docx");

const DOCS = __dirname; // docs/
const SRC = path.join(DOCS, "report.md");
const OUT = process.argv[2] || path.join(DOCS, "Final_Report.docx");
const CONTENT_WIDTH = 9026; // A4 width 11906 minus 2 × 1440 margins (DXA)
const lines = fs.readFileSync(SRC, "utf8").replace(/\r/g, "").split("\n");

// ── inline markdown → runs ───────────────────────────────────────────────────
const TOKEN = /(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|\[[^\]]+\]|\*[^*\s][^*]*?\*)/;
function runs(text, style = {}) {
  const out = [];
  for (const part of text.split(TOKEN)) {
    if (!part) continue;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      out.push(...runs(part.slice(2, -2), { ...style, bold: true }));
    } else if (part.startsWith("`") && part.endsWith("`")) {
      out.push(new TextRun({ ...style, text: part.slice(1, -1), font: "Consolas", size: style.size ? style.size - 2 : 20 }));
    } else if (/^\[[^\]]+\]\([^)]+\)$/.test(part)) {
      const [, label, url] = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      out.push(new ExternalHyperlink({ link: url, children: [new TextRun({ ...style, text: label, style: "Hyperlink" })] }));
    } else if (/^\[\d+(,\s*\d+)*\]$/.test(part)) {
      out.push(new TextRun({ ...style, text: part })); // reference citation like [8]
    } else if (/^\[[^\]]+\]$/.test(part)) {
      out.push(new TextRun({ ...style, text: part, highlight: "yellow" })); // placeholder still to fill in
    } else if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      out.push(...runs(part.slice(1, -1), { ...style, italics: true }));
    } else {
      out.push(new TextRun({ ...style, text: part }));
    }
  }
  return out;
}

// ── tables ───────────────────────────────────────────────────────────────────
const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
function table(rows, { header = true, fontSize = 19 } = {}) {
  const ncol = rows[0].length;
  const weight = Array.from({ length: ncol }, (_, i) =>
    Math.max(6, Math.min(60, ...rows.map((r) => (r[i] || "").replace(/\*\*|`/g, "").length))));
  const total = weight.reduce((a, b) => a + b, 0);
  const widths = weight.map((w) => Math.floor((w / total) * CONTENT_WIDTH));
  widths[widths.length - 1] += CONTENT_WIDTH - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    layout: TableLayoutType.FIXED,
    columnWidths: widths,
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: header && ri === 0,
      children: widths.map((w, ci) => new TableCell({
        width: { size: w, type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        shading: header && ri === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "DCE6F2" } : undefined,
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({ spacing: { before: 0, after: 0 },
          children: runs(r[ci] || "", { size: fontSize, bold: header && ri === 0 ? true : undefined }) })],
      })),
    })),
  });
}

// ── image (PNG width/height from the header) ─────────────────────────────────
function image(file, alt) {
  const data = fs.readFileSync(file);
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  const widthPx = 600; // ~6.25 in at 96 dpi, the text column
  // keepNext: Word keeps the figure on the same page as its caption
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { title: alt.split(":")[0], description: alt, name: path.basename(file) } })] });
}

// ── title page (lines before "## Abstract") ──────────────────────────────────
const body = [];
const abstractAt = lines.findIndex((l) => l.startsWith("## Abstract"));
const head = lines.slice(0, abstractAt);
const title = head.find((l) => l.startsWith("# ")).slice(2);
const bolds = head.filter((l) => /^\*\*.+\*\*$/.test(l)).map((l) => l.slice(2, -2));
const infoRows = head.filter((l) => l.startsWith("|") && !/^\|\s*-/.test(l)).map(cells).filter((r) => r.some((c) => c));
body.push(
  new Paragraph({ spacing: { before: 2200, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: title, bold: true, size: 52, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: bolds[0], size: 28, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: bolds[1], bold: true, size: 26 })] }),
  table(infoRows, { header: false, fontSize: 22 }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Contents", bold: true, size: 32, color: "1F3864" })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ spacing: { before: 120 }, children: [new TextRun({
    text: "Right-click the table of contents and choose Update Field to fill in the page numbers.",
    italics: true, size: 18, color: "7F7F7F" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ── body ─────────────────────────────────────────────────────────────────────
let i = abstractAt;
let listInstance = 0;
const isBlank = (l) => l.trim() === "";
while (i < lines.length) {
  const line = lines[i];
  if (isBlank(line)) { i++; continue; }

  if (line.startsWith("## ")) {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(line.slice(3)) })); i++; continue;
  }
  if (line.startsWith("### ")) {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(line.slice(4)) })); i++; continue;
  }
  if (line.startsWith("|")) {
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) { if (!/^\|\s*-/.test(lines[i])) rows.push(cells(lines[i])); i++; }
    body.push(table(rows), new Paragraph({ spacing: { after: 0 }, children: [] }));
    continue;
  }
  if (line.startsWith("![")) {
    let block = line;
    while (!/\]\([^)]+\)\s*$/.test(block)) block += " " + lines[++i];
    i++;
    const [, alt, src] = block.match(/^!\[([\s\S]*)\]\(([^)]+)\)\s*$/);
    body.push(image(path.join(DOCS, src), alt.replace(/\s+/g, " ").trim()));
    continue;
  }
  if (line.startsWith(">")) {
    let text = "";
    while (i < lines.length && lines[i].startsWith(">")) { text += " " + lines[i].replace(/^>\s?/, ""); i++; }
    body.push(new Paragraph({ spacing: { before: 120, after: 120 },
      children: runs(text.trim(), { italics: true, color: "595959", size: 20 }) }));
    continue;
  }
  if (/^- /.test(line) || /^\d+\. /.test(line)) {
    const numbered = /^\d+\. /.test(line);
    listInstance++;
    while (i < lines.length && (/^- /.test(lines[i]) || /^\d+\. /.test(lines[i]))) {
      let text = lines[i].replace(/^- |^\d+\. /, ""); i++;
      while (i < lines.length && /^\s{2,}\S/.test(lines[i])) { text += " " + lines[i].trim(); i++; }
      body.push(new Paragraph({ numbering: { reference: numbered ? "numbers" : "bullets", level: 0, instance: listInstance },
        spacing: { after: 80 }, children: runs(text) }));
    }
    continue;
  }
  // paragraph (consecutive non-blank lines)
  let text = line; i++;
  while (i < lines.length && !isBlank(lines[i]) && !/^(#|\||>|!\[|- |\d+\. )/.test(lines[i])) { text += " " + lines[i]; i++; }
  const caption = /^\*(Table|Figure) \d+\./.test(text);
  body.push(caption
    ? new Paragraph({ spacing: { before: 60, after: 240 }, children: runs(text, { size: 18, color: "404040" }) })
    : new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.JUSTIFIED, children: runs(text) }));
}

// ── document ─────────────────────────────────────────────────────────────────
const doc = new Document({
  creator: "Sem Chitra, Sar Sakal, Earn Pisey, Sam Reaksmey",
  title: "Khmer Speech-to-Subtitles — Final Project Report",
  styles: {
    default: { document: { run: { font: { ascii: "Calibri", hAnsi: "Calibri", eastAsia: "Calibri", cs: "Khmer UI" }, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E5395" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "7F7F7F" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
