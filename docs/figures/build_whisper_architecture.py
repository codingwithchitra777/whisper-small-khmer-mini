"""Build the Whisper architecture figure (Figure 1 in docs/report.md): SVG, HTML page and PNG.

    python docs/figures/build_whisper_architecture.py      # PNG needs Google Chrome installed

Whisper-small as used in this project (Radford et al., 2023): encoder–decoder Transformer, 12 + 12 blocks,
d_model 768, 12 attention heads, 51,865-token vocabulary, 448-token decoder limit.
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent

PAPER, INK, MUTED, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#7a8399"
ACCENT, ACCENT_TINT, LINK = "#eb6c36", "rgba(235,108,54,0.10)", "#2e5aa8"
SANS, MONO = "'Geist', 'Segoe UI', sans-serif", "'Geist Mono', Consolas, monospace"
W, H, TOP = 1040, 560, 40
NODE_W, NODE_H = 280, 72
LEFT_X, RIGHT_X = 80, 680
ROWS = (96, 208, 320, 432)

STYLES = {
    "focal": (ACCENT_TINT, ACCENT, ""),
    "step": ("#ffffff", INK, ""),
    "input": ("rgba(79,93,117,0.10)", SOFT, ""),
}


def node(x, y, kind, tag, name, sub):
    fill, stroke, dash = STYLES[kind]
    tag_w = 8 + 6 * len(tag)
    cx = x + NODE_W // 2
    return f"""
  <rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="6" fill="{PAPER}"/>
  <rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>
  <rect x="{x + 8}" y="{y + 8}" width="{tag_w}" height="12" rx="2" fill="none" stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>
  <text x="{x + 8 + tag_w / 2}" y="{y + 17}" fill="{stroke}" fill-opacity="0.8" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{tag}</text>
  <text x="{cx}" y="{y + 44}" fill="{INK}" font-size="12" font-weight="600" font-family="{SANS}" text-anchor="middle">{name}</text>
  <text x="{cx}" y="{y + 60}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{sub}</text>"""


def label(cx, y, w, text):
    """Masked arrow label; y is the mask's top edge."""
    return f"""
  <rect x="{cx - w / 2}" y="{y}" width="{w}" height="12" rx="2" fill="{PAPER}"/>
  <text x="{cx}" y="{y + 9}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{text}</text>"""


def down(cx, y_from, y_to, color=MUTED, marker="w-arrow"):
    return f'  <line x1="{cx}" y1="{y_from}" x2="{cx}" y2="{y_to - 2}" stroke="{color}" stroke-width="1.2" marker-end="url(#{marker})"/>\n'


LC, RC = LEFT_X + NODE_W // 2, RIGHT_X + NODE_W // 2
zones = f"""
  <rect x="56" y="64" width="328" height="352" rx="8" fill="rgba(45,49,66,0.03)" stroke="rgba(45,49,66,0.25)" stroke-width="1"/>
  <text x="72" y="84" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">ENCODER · reads the audio once</text>
  <rect x="656" y="64" width="360" height="464" rx="8" fill="rgba(45,49,66,0.03)" stroke="rgba(45,49,66,0.25)" stroke-width="1"/>
  <text x="672" y="84" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">DECODER · writes one token at a time</text>"""

arrows = (
    down(LC, ROWS[0] + NODE_H, ROWS[1]) + down(LC, ROWS[1] + NODE_H, ROWS[2])
    + down(RC, ROWS[0] + NODE_H, ROWS[1]) + down(RC, ROWS[1] + NODE_H, ROWS[2])
    + down(RC, ROWS[2] + NODE_H, ROWS[3], ACCENT, "w-arrow-accent")
    # encoder states feed every decoder block through cross-attention
    + f'  <line x1="{LEFT_X + NODE_W}" y1="{ROWS[2] + 36}" x2="{RIGHT_X - 2}" y2="{ROWS[2] + 36}" stroke="{LINK}" stroke-width="1.2" marker-end="url(#w-arrow-link)"/>\n'
    # autoregressive loop: each predicted token is appended to the decoder input
    + f'  <path d="M{RIGHT_X + NODE_W},{ROWS[3] + 36} H992 Q1000,{ROWS[3] + 36} 1000,{ROWS[3] + 28} V{ROWS[1] + 44} Q1000,{ROWS[1] + 36} 992,{ROWS[1] + 36} H{RIGHT_X + NODE_W + 2}" fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4" marker-end="url(#w-arrow)"/>\n'
    + label(520, ROWS[2] + 16, 112, "ENCODER STATES")
    + label(520, ROWS[2] + 44, 112, "CROSS-ATTENTION")
    + label(962, ROWS[3] - 26, 68, "NEXT TOKEN")
)

nodes = (
    node(LEFT_X, ROWS[0], "input", "IN", "Audio chunk", "16 kHz mono · ≤ 30 s (we use ≤ 10 s)")
    + node(LEFT_X, ROWS[1], "step", "STEP", "Log-Mel spectrogram", "80 channels × 3,000 frames")
    + node(LEFT_X, ROWS[2], "step", "ENC", "Encoder × 12 blocks", "conv stem · self-attention + MLP · d=768")
    + node(RIGHT_X, ROWS[0], "focal", "PROMPT", "Khmer task tokens", "&lt;|sot|&gt; &lt;|km|&gt; &lt;|transcribe|&gt; &lt;|0.00|&gt;")
    + node(RIGHT_X, ROWS[1], "step", "EMB", "Token + position embedding", "51,865-token vocabulary")
    + node(RIGHT_X, ROWS[2], "step", "DEC", "Decoder × 12 blocks", "masked self-attn · cross-attn · MLP")
    + node(RIGHT_X, ROWS[3], "focal", "OUT", "Khmer text + timestamps", "ភាសាខ្មែរ… &lt;|8.68|&gt; · ≤ 448 tokens")
)


def legend_item(x, swatch, text):
    return f'{swatch}<text x="{x + 28}" y="579" fill="{MUTED}" font-size="9" font-family="{MONO}">{text}</text>'


legend = f"""
  <line x1="24" y1="552" x2="{W - 24}" y2="552" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>
  <text x="24" y="579" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.14em">LEGEND</text>
""" + legend_item(96, f'<rect x="96" y="568" width="20" height="16" rx="3" fill="{ACCENT_TINT}" stroke="{ACCENT}"/>', "set by our fine-tuning (Khmer + timestamps)") \
    + legend_item(392, f'<rect x="392" y="568" width="20" height="16" rx="3" fill="#ffffff" stroke="{INK}"/>', "Whisper-small layer (all 241M weights fine-tuned)") \
    + legend_item(720, f'<line x1="720" y1="576" x2="740" y2="576" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5,4"/>', "autoregressive loop")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {TOP} {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="whisper-architecture-title whisper-architecture-desc">
  <title id="whisper-architecture-title">Whisper-small encoder–decoder architecture</title>
  <desc id="whisper-architecture-desc">The encoder turns up to 30 seconds of audio, as an 80-channel log-Mel spectrogram, into hidden states through 12 Transformer blocks; the decoder, prompted with Khmer transcription tokens, attends to those states through cross-attention in its 12 blocks and writes Khmer text with timestamp tokens one token at a time, up to 448 tokens.</desc>
  <defs>
    <style>@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;600&amp;family=Geist+Mono:wght@400;500&amp;display=swap');</style>
    <marker id="w-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>
    <marker id="w-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/></marker>
    <marker id="w-arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>
  </defs>
  <rect x="0" y="{TOP}" width="{W}" height="{H}" fill="{PAPER}"/>
{zones}
{arrows}
{nodes}
{legend}
</svg>
"""

(OUT / "whisper-architecture.svg").write_text(svg, encoding="utf-8")
render = OUT / "_render_whisper.html"
render.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0">{svg}</body></html>', encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                "--force-device-scale-factor=2", "--virtual-time-budget=5000",
                f"--screenshot={OUT / 'whisper-architecture.png'}", render.as_uri()], check=True, capture_output=True)
render.unlink()
print("wrote whisper-architecture.svg and .png")
