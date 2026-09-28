"use client";

interface YoutubeEmbedProps {
  videoId: string;
}

/**
 * Extract the YouTube video ID from a full URL.
 * Supports:
 *   - https://www.youtube.com/watch?v=VIDEO_ID
 *   - https://youtu.be/VIDEO_ID
 *   - https://www.youtube.com/embed/VIDEO_ID
 */
export function extractVideoId(url: string): string | null {
  try {
    const u = new URL(url);
    // Short link
    if (u.hostname.includes("youtu.be")) {
      return u.pathname.slice(1).split("?")[0] || null;
    }
    // Embed URL
    if (u.pathname.startsWith("/embed/")) {
      return u.pathname.replace("/embed/", "").split("?")[0] || null;
    }
    // Standard watch URL
    return u.searchParams.get("v");
  } catch {
    return null;
  }
}

export default function YoutubeEmbed({ videoId }: YoutubeEmbedProps) {
  return (
    <div className="glass-card embed-section">
      <div className="embed-header">
        <span className="embed-title">🎬 Video Preview</span>
      </div>
      <div className="embed-wrapper">
        <iframe
          src={`https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1`}
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          loading="lazy"
        />
      </div>
    </div>
  );
}
