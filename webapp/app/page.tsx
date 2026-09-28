"use client";

import { useState, useRef } from "react";
import ProgressSteps, { type Step } from "../components/ProgressSteps";
import SubtitleTimeline, { type Segment } from "../components/SubtitleTimeline";
import YoutubeEmbed, { extractVideoId } from "../components/YoutubeEmbed";
import ExportButton from "../components/ExportButton";

// ── Helper: format seconds → human-readable ──────────────────────────────────
function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}m ${s}s`;
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [url, setUrl]           = useState("");
  const [step, setStep]         = useState<Step>("idle");
  const [segments, setSegments] = useState<Segment[]>([]);
  const [videoId, setVideoId]   = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [error, setError]       = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;

    // Reset state
    setError(null);
    setSegments([]);
    setDuration(null);

    // Extract and show YouTube embed immediately
    const vid = extractVideoId(trimmed);
    setVideoId(vid);

    // Step 1 — Downloading
    setStep("downloading");

    try {
      // Steps are simulated on the frontend while the backend does real work
      // Backend combines download + extract + transcribe into one call
      const progressTimeout1 = setTimeout(() => setStep("extracting"),   3_000);
      const progressTimeout2 = setTimeout(() => setStep("transcribing"), 7_000);

      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
      });

      clearTimeout(progressTimeout1);
      clearTimeout(progressTimeout2);

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `Server error (${res.status})`);
      }

      const data = await res.json();

      if (!data.success) {
        throw new Error(data.error ?? "Transcription failed");
      }

      setSegments(data.segments ?? []);
      setDuration(data.duration_seconds ?? null);
      setStep("done");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error occurred";
      setError(msg);
      setStep("error");
    }
  };

  const handleReset = () => {
    setUrl("");
    setStep("idle");
    setSegments([]);
    setVideoId(null);
    setDuration(null);
    setError(null);
    inputRef.current?.focus();
  };

  const isLoading = step === "downloading" || step === "extracting" || step === "transcribing";

  return (
    <div className="main-content">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="hero" aria-labelledby="hero-title">
        <p className="hero-eyebrow">Khmer ASR · Whisper Fine-tuned</p>
        <h1 id="hero-title" className="hero-title">
          Extract{" "}
          <span className="hero-title-gradient">Khmer Subtitles</span>
          <br />
          from YouTube
        </h1>
        <p className="hero-sub">
          Paste any YouTube URL. Our fine-tuned Whisper model will download
          the audio and generate timestamped Khmer subtitles — ready to export
          as&nbsp;.SRT.
        </p>
        <p className="hero-khmer" lang="km">
          ស្រង់អត្ថបទភាសាខ្មែរពីវីដេអូ YouTube ដោយស្វ័យប្រវត្តិ
        </p>
      </section>

      {/* ── URL Input ────────────────────────────────────────────────────── */}
      <form onSubmit={handleSubmit} aria-label="YouTube URL submission form">
        <div className="glass-card input-section">
          <label className="input-label" htmlFor="youtube-url">
            YouTube URL
          </label>
          <div className="input-row">
            <input
              ref={inputRef}
              id="youtube-url"
              type="url"
              className="url-input"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={isLoading}
              aria-required="true"
              aria-describedby="url-hint"
            />
            {step !== "idle" && !isLoading ? (
              <button
                type="button"
                id="reset-btn"
                className="btn-secondary"
                onClick={handleReset}
              >
                🔄 New
              </button>
            ) : null}
            <button
              type="submit"
              id="extract-btn"
              className="btn-primary"
              disabled={isLoading || !url.trim()}
              aria-busy={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Processing…
                </>
              ) : (
                <>🎙️ Extract</>
              )}
            </button>
          </div>
          <p id="url-hint" style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: "0.6rem" }}>
            Works with any public YouTube video. Processing time depends on video length.
          </p>
        </div>
      </form>

      {/* ── Progress ─────────────────────────────────────────────────────── */}
      <ProgressSteps step={step} />

      {/* ── Error ────────────────────────────────────────────────────────── */}
      {step === "error" && error && (
        <div className="error-card" role="alert" aria-live="assertive">
          <span className="error-icon" aria-hidden="true">⚠️</span>
          <div>
            <p className="error-title">Something went wrong</p>
            <p className="error-msg">{error}</p>
          </div>
        </div>
      )}

      {/* ── Stats ────────────────────────────────────────────────────────── */}
      {step === "done" && segments.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
          <div className="stats-bar">
            <div className="stat-item">
              <span className="stat-value">{segments.length}</span>
              <span className="stat-label">Segments</span>
            </div>
            {duration != null && (
              <div className="stat-item">
                <span className="stat-value">{formatDuration(duration)}</span>
                <span className="stat-label">Duration</span>
              </div>
            )}
            <div className="stat-item">
              <span className="stat-value" lang="km">ខ្មែរ</span>
              <span className="stat-label">Language</span>
            </div>
          </div>
          <ExportButton segments={segments} videoUrl={url} />
        </div>
      )}

      {/* ── Results Grid ─────────────────────────────────────────────────── */}
      {(videoId || segments.length > 0) && (
        <div className="results-grid">
          {/* Left — Subtitles */}
          <SubtitleTimeline segments={segments} />

          {/* Right — YouTube Embed */}
          {videoId && <YoutubeEmbed videoId={videoId} />}
        </div>
      )}

      {/* ── Empty landing state ───────────────────────────────────────────── */}
      {step === "idle" && (
        <div className="glass-card" style={{ marginTop: "1rem" }}>
          <div className="empty-state">
            <div className="empty-icon">🇰🇭</div>
            <p className="empty-title">Ready to transcribe Khmer audio</p>
            <p className="empty-sub">
              Powered by a fine-tuned{" "}
              <span style={{ color: "var(--accent-end)" }}>OpenAI Whisper</span>{" "}
              model trained on Khmer speech data
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
