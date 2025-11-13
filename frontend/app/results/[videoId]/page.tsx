"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

interface Frame {
  frame_id: string;
  video_id: string;
  index: number;
  url: string;
  width: number;
  height: number;
  created_at: string;
  event_label: string | null;
  event_class: number | null;
  swing_metrics: Record<string, number | null> | null;
}

interface VideoStatus {
  video_id: string;
  status: string;
  s3_key: string;
  frame_urls: string[];
}

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const videoId = params.videoId as string;

  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(true);

  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  // Poll video status until processing is complete
  useEffect(() => {
    if (!videoId || !polling) return;

    const pollStatus = async () => {
      try {
        const response = await fetch(`${apiBase}/videos/${videoId}`);
        if (!response.ok) {
          if (response.status === 404) {
            setError("Video not found");
            setPolling(false);
            setLoading(false);
            return;
          }
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data: VideoStatus = await response.json();
        setVideoStatus(data);

        if (data.status === "processed") {
          setPolling(false);
          // Fetch frames
          await fetchFrames();
        } else if (data.status === "failed") {
          setPolling(false);
          setError("Video processing failed");
          setLoading(false);
        } else {
          // Continue polling
          setTimeout(pollStatus, 2500); // Poll every 2.5 seconds
        }
      } catch (err) {
        console.error("Error polling video status:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
        setPolling(false);
        setLoading(false);
      }
    };

    pollStatus();
  }, [videoId, polling, apiBase]);

  const fetchFrames = async () => {
    try {
      const response = await fetch(`${apiBase}/videos/${videoId}/frames`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      // Sort frames by event_class to ensure correct order: Address (0), Top (3), Mid-downswing (4), Impact (5), Finish (7)
      const sortedFrames = (data.frames || []).sort((a: Frame, b: Frame) => {
        const eventClassOrder = [0, 3, 4, 5, 7]; // Address, Top, Mid-downswing, Impact, Finish
        const aIndex = a.event_class !== null ? eventClassOrder.indexOf(a.event_class) : 999;
        const bIndex = b.event_class !== null ? eventClassOrder.indexOf(b.event_class) : 999;
        return aIndex - bIndex;
      });
      
      // Add cache-busting parameter to image URLs to prevent browser caching issues
      // Use frame_id to ensure each frame has a unique cache-busting parameter
      const framesWithCacheBust = sortedFrames.map((frame: Frame) => ({
        ...frame,
        url: `${frame.url}${frame.url.includes('?') ? '&' : '?'}cb=${frame.frame_id}&t=${Date.now()}`,
      }));
      
      console.log('Fetched frames:', framesWithCacheBust.map((f: Frame) => ({
        event_class: f.event_class,
        event_label: f.event_label,
        url: f.url.substring(0, 100) + '...',
      })));
      
      setFrames(framesWithCacheBust);
      setLoading(false);
    } catch (err) {
      console.error("Error fetching frames:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch frames");
      setLoading(false);
    }
  };

  const formatMetrics = (): string => {
    if (frames.length === 0) return "No metrics available";

    // Map event labels to display names and position keys
    const eventLabelMap: Record<string, { displayName: string; positionKey: string }> = {
      "Address": { displayName: "Address", positionKey: "address" },
      "Top": { displayName: "Top", positionKey: "top" },
      "Mid-downswing (arm parallel)": { displayName: "Mid-downswing", positionKey: "mid_ds" },
      "Impact": { displayName: "Impact", positionKey: "impact" },
      "Finish": { displayName: "Finish", positionKey: "finish" },
    };

    // Order for display
    const positionOrder = ["Address", "Top", "Mid-downswing (arm parallel)", "Impact", "Finish"];

    let output = "";
    for (const eventLabel of positionOrder) {
      const frame = frames.find((f) => f.event_label === eventLabel);
      if (frame && frame.swing_metrics) {
        const { displayName } = eventLabelMap[eventLabel] || { displayName: eventLabel };
        output += `${displayName}:\n`;
        const metrics = frame.swing_metrics;
        for (const [key, value] of Object.entries(metrics)) {
          if (value !== null && value !== undefined) {
            // Format metric name (remove _deg suffix for display, add ° symbol)
            const displayKey = key.replace(/_deg$/, "").replace(/_/g, " ");
            const displayValue =
              key.endsWith("_deg") || key.includes("angle")
                ? `${value.toFixed(1)}°`
                : typeof value === "number"
                ? value.toFixed(2)
                : value;
            output += `  - ${displayKey}: ${displayValue}\n`;
          }
        }
        output += "\n";
      }
    }

    return output || "No metrics available";
  };

  if (loading && polling) {
    return (
      <main style={{ padding: "2rem", textAlign: "center" }}>
        <h1>Processing Video...</h1>
        <p>Please wait while we extract key frames and compute metrics.</p>
        <div style={{ marginTop: "2rem" }}>
          <div
            style={{
              display: "inline-block",
              width: "40px",
              height: "40px",
              border: "4px solid #f3f3f3",
              borderTop: "4px solid #3498db",
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
            }}
          />
        </div>
        <style jsx>{`
          @keyframes spin {
            0% {
              transform: rotate(0deg);
            }
            100% {
              transform: rotate(360deg);
            }
          }
        `}</style>
      </main>
    );
  }

  if (error) {
    return (
      <main style={{ padding: "2rem", textAlign: "center" }}>
        <h1 style={{ color: "#e74c3c" }}>Error</h1>
        <p>{error}</p>
        <button
          onClick={() => router.push("/upload")}
          style={{
            marginTop: "1rem",
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            cursor: "pointer",
          }}
        >
          Upload Another Video
        </button>
      </main>
    );
  }

  return (
    <main style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto" }}>
      <div style={{ marginBottom: "2rem" }}>
        <h1>Golf Swing Analysis Results</h1>
        <button
          onClick={() => router.push("/upload")}
          style={{
            marginTop: "1rem",
            padding: "0.5rem 1rem",
            fontSize: "0.9rem",
            cursor: "pointer",
          }}
        >
          Upload Another Video
        </button>
      </div>

      {frames.length > 0 ? (
        <>
          <section style={{ marginBottom: "3rem" }}>
            <h2 style={{ marginBottom: "1.5rem" }}>Key Frames</h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
                gap: "1.5rem",
              }}
            >
              {frames.map((frame) => (
                <div
                  key={frame.frame_id}
                  style={{
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                    overflow: "hidden",
                    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                  }}
                >
                  <div
                    style={{
                      width: "100%",
                      backgroundColor: "#f0f0f0",
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                    }}
                  >
                    <img
                      src={frame.url}
                      alt={frame.event_label || `Frame ${frame.index}`}
                      key={`${frame.frame_id}-${frame.event_class}`}
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                      }}
                      onError={(e) => {
                        console.error(`Failed to load image for ${frame.event_label}:`, frame.url);
                      }}
                    />
                  </div>
                  <div style={{ padding: "1rem" }}>
                    <h3 style={{ margin: 0, fontSize: "1.1rem" }}>
                      {frame.event_label || `Frame ${frame.index}`}
                    </h3>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h2 style={{ marginBottom: "1rem" }}>Swing Metrics</h2>
            <textarea
              readOnly
              value={formatMetrics()}
              style={{
                width: "100%",
                minHeight: "300px",
                padding: "1rem",
                fontSize: "0.95rem",
                fontFamily: "monospace",
                border: "1px solid #ddd",
                borderRadius: "4px",
                backgroundColor: "#f9f9f9",
                resize: "vertical",
                lineHeight: "1.6",
              }}
            />
          </section>
        </>
      ) : (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p>No frames available yet. Processing may still be in progress.</p>
        </div>
      )}
    </main>
  );
}

