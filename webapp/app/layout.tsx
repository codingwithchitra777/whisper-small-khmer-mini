import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Khmer Subtitle Extractor — ក្រុមការស្រង់អត្ថបទ",
  description:
    "Extract timestamped Khmer subtitles from YouTube videos using a fine-tuned Whisper ASR model.",
  keywords: ["Khmer", "ASR", "subtitles", "speech recognition", "YouTube", "Whisper"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="km">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <div className="page-wrapper">
          {/* Background ambient blobs */}
          <div className="bg-blob bg-blob-1" aria-hidden="true" />
          <div className="bg-blob bg-blob-2" aria-hidden="true" />
          <div className="bg-blob bg-blob-3" aria-hidden="true" />

          {/* Navigation */}
          <nav className="nav" role="navigation" aria-label="Main navigation">
            <div className="nav-brand">
              <div className="nav-logo-icon" aria-hidden="true">🎙️</div>
              <span className="nav-title">KhmerSub</span>
            </div>
            <span className="nav-badge">Thesis Project</span>
          </nav>

          {/* Page content */}
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
