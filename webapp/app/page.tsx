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

// ── Backend job (see api/main.py: POST /jobs, GET /jobs/{id}) ─────────────────
interface Job {
  job_id: string;
  status: "queued" | "downloading" | "transcribing" | "done" | "error";
  queue_position: number;
  chunks_done: number;
  chunks_total: number;
  duration_seconds: number | null;
  segments: Segment[];
  error: string | null;
}

const POLL_MS = 2_000;
const MAX_POLL_FAILURES = 5; // tolerate brief network blips before giving up

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function readJson(res: Response) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `Server error (${res.status})`);
  return data;
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [url, setUrl]           = useState("");
  const [step, setStep]         = useState<Step>("idle");
  const [detail, setDetail]     = useState<string | null>(null);
  const [fraction, setFraction] = useState<number | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [videoId, setVideoId]   = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [error, setError]       = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  // Bumped on every submit/reset so a stale polling loop stops updating the page.
  const runRef = useRef(0);

  // Map the backend's job status onto the progress steps.
  const showJob = (job: Job) => {
    if (job.status === "queued") {
      setStep("downloading");
      setDetail(job.queue_position > 0 ? `Waiting in queue (${job.queue_position} ahead)…` : "Starting…");
    } else if (job.status === "downloading") {
      setStep("downloading");
      setDetail("Downloading audio from YouTube…");
    } else if (job.status === "transcribing") {
      if (job.duration_seconds != null) setDuration(job.duration_seconds);
      if (job.chunks_total === 0) {
        setStep("extracting");
        setDetail("Splitting the audio at pauses…");
      } else {
        setStep("transcribing");
        setFraction(job.chunks_done / job.chunks_total);
        setDetail(`Transcribing ${job.chunks_done} / ${job.chunks_total} chunks`);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;

    const run = ++runRef.current;
    setError(null);
    setSegments([]);
    setDuration(null);
    setFraction(null);
    setVideoId(extractVideoId(trimmed)); // show the video right away
    setStep("downloading");
    setDetail("Starting…");

    try {
      // The backend works in the background; each request here returns quickly, so long
      // videos never hit a proxy's request time limit (Cloudflare: 100 s).
      let job: Job = await readJson(
        await fetch("/api/jobs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: trimmed }),
        })
      );

      let failures = 0;
      while (job.status !== "done" && job.status !== "error") {
        if (run !== runRef.current) return;
        showJob(job);
        await sleep(POLL_MS);
        try {
          job = await readJson(await fetch(`/api/jobs/${job.job_id}`, { cache: "no-store" }));
          failures = 0;
        } catch (pollError) {
          if (++failures >= MAX_POLL_FAILURES) throw pollError;
        }
      }
      if (run !== runRef.current) return;

      if (job.status === "error") throw new Error(job.error ?? "Transcription failed");
      setSegments(job.segments ?? []);
      setDuration(job.duration_seconds ?? null);
      setFraction(null);
      setDetail(null);
      setStep("done");
    } catch (err: unknown) {
      if (run !== runRef.current) return;
      setError(err instanceof Error ? err.message : "Unknown error occurred");
      setDetail(null);
      setFraction(null);
      setStep("error");
    }
  };

  const handleReset = () => {
    runRef.current++;
    setDetail(null);
    setFraction(null);
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
      <ProgressSteps step={step} detail={detail} fraction={fraction} />

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
