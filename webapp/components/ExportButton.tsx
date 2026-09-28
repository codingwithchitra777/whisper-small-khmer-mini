"use client";

import type { Segment } from "./SubtitleTimeline";

interface ExportButtonProps {
  segments: Segment[];
  videoUrl?: string;
}

/** Format seconds → SRT timestamp: HH:MM:SS,mmm */
function toSrtTime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")},${String(ms).padStart(3,"0")}`;
}

function buildSRT(segments: Segment[]): string {
  return segments
    .map((seg, i) => {
      const start = toSrtTime(seg.start);
      const end   = toSrtTime(seg.end);
      return `${i + 1}\n${start} --> ${end}\n${seg.text}\n`;
    })
    .join("\n");
}

function buildTXT(segments: Segment[]): string {
  return segments.map(s => `[${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s] ${s.text}`).join("\n");
}

function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob(["\uFEFF" + content], { type: mime }); // BOM for UTF-8 Khmer
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ExportButton({ segments, videoUrl }: ExportButtonProps) {
  if (!segments || segments.length === 0) return null;

  const stem = "khmer_subtitle";

  return (
    <div className="export-toolbar">
      <button
        id="export-srt-btn"
        className="btn-secondary"
        onClick={() => downloadFile(buildSRT(segments), `${stem}.srt`, "text/plain;charset=utf-8")}
        title="Download as SubRip subtitle file"
      >
        ⬇️ Export SRT
      </button>
      <button
        id="export-txt-btn"
        className="btn-secondary"
        onClick={() => downloadFile(buildTXT(segments), `${stem}.txt`, "text/plain;charset=utf-8")}
        title="Download as plain text with timestamps"
      >
        ⬇️ Export TXT
      </button>
      <button
        id="copy-text-btn"
        className="btn-secondary"
        onClick={() => {
          const plain = segments.map(s => s.text).join("\n");
          navigator.clipboard.writeText(plain);
        }}
        title="Copy all Khmer text to clipboard"
      >
        📋 Copy Text
      </button>
    </div>
  );
}
