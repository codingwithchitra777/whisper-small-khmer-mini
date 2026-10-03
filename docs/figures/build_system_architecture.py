"""Build the system architecture figure (Figure 1 in docs/report.md): SVG, HTML page and PNG.

    python docs/figures/build_system_architecture.py      # PNG needs Google Chrome installed
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#7a8399"
ACCENT, ACCENT_TINT, LINK = "#eb6c36", "rgba(235,108,54,0.10)", "#2e5aa8"
SANS, MONO = "'Geist', 'Segoe UI', sans-serif", "'Geist Mono', Consolas, monospace"
W, H, TOP = 1040, 536, 64

STYLES = {
    "focal": (ACCENT_TINT, ACCENT, ""),
    "step": ("#ffffff", INK, ""),
    "external": ("rgba(45,49,66,0.03)", "rgba(45,49,66,0.30)", ""),
    "input": ("rgba(79,93,117,0.10)", SOFT, ""),
    "optional": ("rgba(45,49,66,0.02)", "rgba(45,49,66,0.20)", ' stroke-dasharray="4,3"'),
}


def node(x, y, kind, tag, name, sub, w=160, h=72):
    fill, stroke, dash = STYLES[kind]
    tag_w = 8 + 6 * len(tag)
    cx = x + w // 2
    return f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{PAPER}"/>
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>
  <rect x="{x + 8}" y="{y + 8}" width="{tag_w}" height="12" rx="2" fill="none" stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>
  <text x="{x + 8 + tag_w / 2}" y="{y + 17}" fill="{stroke}" fill-opacity="0.8" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{tag}</text>
  <text x="{cx}" y="{y + 44}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{name}</text>
  <text x="{cx}" y="{y + 60}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{sub}</text>"""


def label(cx, y, w, text):
    """Masked arrow label; y is the mask's top edge."""
    return f"""
  <rect x="{cx - w / 2}" y="{y}" width="{w}" height="12" rx="2" fill="{PAPER}"/>
  <text x="{cx}" y="{y + 9}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{text}</text>"""


arrows = f"""
  <!-- Browser -> Next.js (request) and back (subtitles) -->
  <line x1="296" y1="112" x2="202" y2="112" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arch-arrow)"/>
  <line x1="200" y1="136" x2="294" y2="136" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#arch-arrow)"/>
  <!-- Next.js -> FastAPI (HTTP proxy) -->
  <line x1="92" y1="160" x2="92" y2="278" stroke="{LINK}" stroke-width="1.2" marker-end="url(#arch-arrow-link)"/>
  <!-- pipeline -->
  <line x1="200" y1="316" x2="238" y2="316" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arch-arrow)"/>
  <line x1="400" y1="316" x2="438" y2="316" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arch-arrow)"/>
  <line x1="600" y1="316" x2="638" y2="316" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arch-arrow)"/>
  <line x1="800" y1="316" x2="838" y2="316" stroke="{ACCENT}" stroke-width="1.2" marker-end="url(#arch-arrow-accent)"/>
  <!-- segments -> Next.js (JSON response) -->
  <path d="M920,280 L920,208 Q920,200 912,200 L156,200 Q148,200 148,192 L148,162" fill="none" stroke="{LINK}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#arch-arrow-link)"/>
  <!-- YouTube -> download; fine-tuning -> model -->
  <line x1="320" y1="456" x2="320" y2="354" stroke="{LINK}" stroke-width="1.2" marker-end="url(#arch-arrow-link)"/>
  <line x1="720" y1="456" x2="720" y2="354" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#arch-arrow)"/>
""" + label(248, 92, 64, "YOUTUBE URL") + label(248, 144, 56, "SUBTITLES") + label(56, 180, 56, "HTTP PROXY") \
    + label(532, 180, 76, "SEGMENTS JSON") + label(360, 412, 64, "AUDIO TRACK") + label(752, 412, 48, "WEIGHTS")

zone = f"""
  <rect x="24" y="232" width="992" height="176" rx="8" fill="rgba(45,49,66,0.03)" stroke="rgba(45,49,66,0.25)" stroke-width="1"/>
  <text x="40" y="396" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">FASTAPI BACK END · api/main.py · :8000</text>"""

nodes = (
    node(40, 88, "step", "WEB", "Next.js web app", "webapp/ · :3000")
    + node(296, 88, "input", "USER", "Viewer's browser", "pastes a YouTube URL")
    + node(40, 280, "step", "API", "Receive request", "POST /process")
    + node(240, 280, "step", "STEP", "Download audio", "yt-dlp + ffmpeg")
    + node(440, 280, "step", "STEP", "Split at pauses", "chunks ≤ 15 s")
    + node(640, 280, "focal", "MODEL", "Whisper-small Khmer", "241M · timestamps")
    + node(840, 280, "step", "OUT", "Subtitle segments", "{start, end, text}")
    + node(240, 456, "external", "EXT", "YouTube", "video audio")
    + node(640, 456, "optional", "JOB", "Fine-tuning (RunPod)", "16 h Khmer · offline")
)


def legend_item(x, swatch, text):
    return f'{swatch}<text x="{x + 28}" y="579" fill="{MUTED}" font-size="9" font-family="{MONO}">{text}</text>'


legend = f"""
  <line x1="24" y1="552" x2="{W - 24}" y2="552" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>
  <text x="24" y="579" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>
""" + legend_item(96, f'<rect x="96" y="568" width="20" height="16" rx="3" fill="{ACCENT_TINT}" stroke="{ACCENT}"/>', "fine-tuned model") \
    + legend_item(252, f'<rect x="252" y="568" width="20" height="16" rx="3" fill="#ffffff" stroke="{INK}"/>', "service / step") \
    + legend_item(400, '<rect x="400" y="568" width="20" height="16" rx="3" fill="rgba(45,49,66,0.03)" stroke="rgba(45,49,66,0.30)"/>', "external") \
    + legend_item(512, '<rect x="512" y="568" width="20" height="16" rx="3" fill="rgba(45,49,66,0.02)" stroke="rgba(45,49,66,0.20)" stroke-dasharray="4,3"/>', "offline") \
    + legend_item(620, f'<line x1="620" y1="576" x2="640" y2="576" stroke="{LINK}" stroke-width="1.2"/>', "HTTP / external data") \
    + legend_item(808, f'<line x1="808" y1="576" x2="828" y2="576" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4"/>', "response / offline")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {TOP} {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="system-architecture-title system-architecture-desc">
  <title id="system-architecture-title">Khmer speech-to-subtitles system architecture</title>
  <desc id="system-architecture-desc">A viewer pastes a YouTube URL into the Next.js web app, which proxies it to the FastAPI back end; the back end downloads the audio, splits it at pauses into chunks of at most 15 seconds, transcribes them with the fine-tuned Whisper-small Khmer model, and returns timestamped subtitle segments to the web app.</desc>
  <defs>
    <style>@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;600&amp;family=Geist+Mono:wght@400;500&amp;display=swap');</style>
    <marker id="arch-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
    <marker id="arch-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
    <marker id="arch-arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>
  </defs>
  <rect x="0" y="{TOP}" width="{W}" height="{H}" fill="{PAPER}"/>
{zone}
{arrows}
{nodes}
{legend}
</svg>
"""

(OUT / "system-architecture.svg").write_text(svg, encoding="utf-8")

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>System architecture</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
 body {{ margin: 0; background: {PAPER}; color: {INK}; font-family: 'Geist', sans-serif; }}
 main {{ max-width: 1088px; margin: 0 auto; padding: 2.5rem 1.5rem; }}
 .eyebrow {{ font-family: 'Geist Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em; color: {MUTED}; margin: 0; }}
 h1 {{ font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.75rem; margin: 0.25rem 0 1.5rem; }}
 svg {{ width: 100%; height: auto; display: block; }}
 footer {{ margin-top: 1.5rem; padding-top: 0.75rem; border-top: 1px solid rgba(45,49,66,0.12);
          font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: {SOFT}; }}
</style></head>
<body><main>
<p class="eyebrow">FIGURE · SECTION 5</p>
<h1>From a YouTube link to Khmer subtitles</h1>
{svg}
<footer>Khmer speech-to-subtitles · whisper-small-khmer-mini · api/main.py + webapp/</footer>
</main></body></html>
"""
(OUT / "system-architecture.html").write_text(page, encoding="utf-8")

# PNG: render the bare SVG at 2x with headless Chrome
render = OUT / "_render.html"
render.write_text(f'<!DOCTYPE html><html><body style="margin:0">{svg}</body></html>', encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                "--force-device-scale-factor=2", "--virtual-time-budget=5000",
                f"--screenshot={OUT / 'system-architecture.png'}", render.as_uri()], check=True, capture_output=True)
render.unlink()
print("wrote", sorted(p.name for p in OUT.iterdir()))
