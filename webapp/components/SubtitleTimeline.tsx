"use client";

export interface Segment {
  start: number;
  end: number;
  text: string;
}

interface SubtitleTimelineProps {
  segments: Segment[];
}

/** Format seconds → MM:SS.ss */
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${String(mins).padStart(2, "0")}:${secs}`;
}

export default function SubtitleTimeline({ segments }: SubtitleTimelineProps) {
  if (!segments || segments.length === 0) {
    return (
      <div className="glass-card subtitle-section">
        <div className="empty-state">
          <div className="empty-icon">🔤</div>
          <p className="empty-title">No subtitles yet</p>
          <p className="empty-sub">Paste a YouTube URL and click Extract to begin</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card subtitle-section">
      <div className="subtitle-header">
        <span className="subtitle-title">📜 Khmer Subtitles</span>
        <span className="subtitle-count">{segments.length} segments</span>
      </div>

      <ol className="subtitle-list" aria-label="Khmer subtitle segments">
        {segments.map((seg, i) => (
          <li
            key={i}
            className="subtitle-item"
            style={{ animationDelay: `${Math.min(i * 40, 600)}ms` }}
          >
            {/* Index */}
            <span className="subtitle-index" aria-hidden="true">
              {String(i + 1).padStart(2, "0")}
            </span>

            {/* Timestamps */}
            <div className="subtitle-timing" aria-label={`From ${formatTime(seg.start)} to ${formatTime(seg.end)}`}>
              <span className="time-badge start">▶ {formatTime(seg.start)}</span>
              <span className="time-badge end">■ {formatTime(seg.end)}</span>
            </div>

            {/* Khmer text */}
            <p className="subtitle-text" lang="km">
              {seg.text}
            </p>
          </li>
        ))}
      </ol>
    </div>
  );
}
