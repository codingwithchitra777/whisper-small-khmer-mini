"use client";

export type Step = "idle" | "downloading" | "extracting" | "transcribing" | "done" | "error";

interface ProgressStepsProps {
  step: Step;
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

export default function ProgressSteps({ step }: ProgressStepsProps) {
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
    </div>
  );
}
