"use client";

export type Step = "idle" | "downloading" | "extracting" | "transcribing" | "done" | "error";

interface ProgressStepsProps {
  step: Step;
  /** Live detail under the steps, e.g. "Transcribing 12 / 40 chunks". */
  detail?: string | null;
  /** 0–1 while transcribing; draws a progress bar. */
  fraction?: number | null;
}

const STEPS: { key: Step; icon: string; label: string }[] = [
  { key: "downloading",  icon: "⬇️", label: "Download" },
  { key: "extracting",   icon: "🎵", label: "Extract Audio" },
  { key: "transcribing", icon: "📝", label: "Transcribe" },
  { key: "done",         icon: "✅", label: "Done" },
];

const STEP_ORDER: Step[] = ["downloading", "extracting", "transcribing", "done"];

function getStepStatus(stepKey: Step, currentStep: Step): "idle" | "active" | "done" {
  if (currentStep === "idle" || currentStep === "error") return "idle";
  const currentIdx = STEP_ORDER.indexOf(currentStep);
  const thisIdx = STEP_ORDER.indexOf(stepKey);
  if (thisIdx < currentIdx) return "done";
  if (thisIdx === currentIdx) return "active";
  return "idle";
}

export default function ProgressSteps({ step, detail, fraction }: ProgressStepsProps) {
  if (step === "idle") return null;

  return (
    <div className="glass-card progress-section" role="status" aria-live="polite">
      <div className="progress-steps">
        {STEPS.map(({ key, icon, label }) => {
          const status = getStepStatus(key, step);
          return (
            <div key={key} className={`progress-step ${status}`}>
              <div className="step-icon">
                {status === "done" ? "✓" : icon}
              </div>
              <span className="step-label">{label}</span>
            </div>
          );
        })}
      </div>
      {fraction != null && step === "transcribing" && (
        <div
          role="progressbar"
          aria-label="Transcription progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(fraction * 100)}
          style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.08)", marginTop: "1rem", overflow: "hidden" }}
        >
          <div
            style={{
              width: `${Math.round(fraction * 100)}%`,
              height: "100%",
              background: "var(--accent-gradient)",
              transition: "width 0.4s ease",
            }}
          />
        </div>
      )}
      {detail && step !== "done" && (
        <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.6rem", textAlign: "center" }}>
          {detail}
        </p>
      )}
    </div>
  );
}
