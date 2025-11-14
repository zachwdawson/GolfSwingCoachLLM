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
  ball_shape: string | null;
  contact: string | null;
  description: string | null;
}

interface SwingFlaw {
  id: string;
  title: string;
  level: string | null;
  contact: string | null;
  ball_shape: string | null;
  cues: string[];
  drills: Array<{ [key: string]: string }>;
  similarity: number;
}

interface VideoProcessResponse {
  video_id: string;
  status: string;
  frames: Frame[];
  metrics: Record<string, Record<string, any>>;
  swing_flaws: SwingFlaw[];
  ball_shape: string | null;
  contact: string | null;
  description: string | null;
  practice_plan: string | null;
}

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const videoId = params.videoId as string;

  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null);
  const [frames, setFrames] = useState<Frame[]>([]);
  const [swingFlaws, setSwingFlaws] = useState<SwingFlaw[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [polling, setPolling] = useState(true);
  const [videoProcessResponse, setVideoProcessResponse] = useState<VideoProcessResponse | null>(null);
  const [metrics, setMetrics] = useState<Record<string, Record<string, any>>>({});
  const [practicePlan, setPracticePlan] = useState<string | null>(null);

  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  // Check localStorage for stored VideoProcessResponse
  useEffect(() => {
    if (!videoId) return;
    
    const storedResponse = localStorage.getItem(`videoResponse_${videoId}`);
    if (storedResponse) {
      try {
        const parsed = JSON.parse(storedResponse);
        setVideoProcessResponse(parsed);
        if (parsed.practice_plan) {
          setPracticePlan(parsed.practice_plan);
        }
        // Clean up localStorage after reading
        localStorage.removeItem(`videoResponse_${videoId}`);
      } catch (e) {
        console.error("Error parsing stored video response:", e);
      }
    }
  }, [videoId]);

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
      setSwingFlaws(data.swing_flaws || []);
      
      // Fetch metrics
      await fetchMetrics();
      
      // Check if practice plan is in the response (for direct upload flow)
      if (data.practice_plan) {
        setPracticePlan(data.practice_plan);
      }
      
      setLoading(false);
    } catch (err) {
      console.error("Error fetching frames:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch frames");
      setLoading(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${apiBase}/videos/${videoId}/metrics`);
      if (response.ok) {
        const data = await response.json();
        setMetrics(data.metrics || {});
      }
    } catch (err) {
      console.error("Error fetching metrics:", err);
    }
  };

  // Construct VideoProcessResponse when all data is available
  useEffect(() => {
    if (!videoProcessResponse && frames.length > 0 && Object.keys(metrics).length > 0 && videoStatus) {
      // Construct VideoProcessResponse from fetched data
      const constructed: VideoProcessResponse = {
        video_id: videoId,
        status: videoStatus.status,
        frames: frames,
        metrics: metrics,
        swing_flaws: swingFlaws,
        ball_shape: videoStatus.ball_shape,
        contact: videoStatus.contact,
        description: videoStatus.description,
        practice_plan: practicePlan,
      };
      setVideoProcessResponse(constructed);
    }
  }, [frames, metrics, swingFlaws, videoStatus, videoId, videoProcessResponse, practicePlan]);

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

  const formatSwingFlaws = (): string => {
    if (swingFlaws.length === 0) return "No swing flaws identified.";

    let output = "";
    swingFlaws.forEach((flaw, index) => {
      output += `${index + 1}. ${flaw.title}\n`;
      output += `   Similarity: ${(flaw.similarity * 100).toFixed(1)}%\n`;
      
      if (flaw.cues && flaw.cues.length > 0) {
        output += `   Cues:\n`;
        flaw.cues.forEach((cue) => {
          output += `     - ${cue}\n`;
        });
      }
      
      if (flaw.drills && flaw.drills.length > 0) {
        output += `   Drills:\n`;
        flaw.drills.forEach((drill) => {
          if (drill["drill explanation"]) {
            output += `     - ${drill["drill explanation"]}\n`;
            if (drill["drill video"]) {
              output += `       Video: ${drill["drill video"]}\n`;
            }
          }
        });
      }
      
      output += "\n";
    });

    return output;
  };

  const renderMarkdown = (text: string): string => {
    if (!text) return "";
    
    let html = text;
    
    // Escape HTML first to prevent XSS
    html = html
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    
    // Split into lines for better processing
    const lines = html.split("\n");
    const processedLines: string[] = [];
    let inList = false;
    let listItems: string[] = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Check for headers
      if (line.match(/^### /)) {
        if (inList) {
          processedLines.push(`<ul style='margin-top: 0.5rem; margin-bottom: 1rem; padding-left: 0; list-style-type: disc;'>${listItems.join("")}</ul>`);
          listItems = [];
          inList = false;
        }
        processedLines.push(`<h3 style='margin-top: 1.5rem; margin-bottom: 0.75rem; font-size: 1.3rem; font-weight: 600;'>${line.replace(/^### /, "")}</h3>`);
      } else if (line.match(/^## /)) {
        if (inList) {
          processedLines.push(`<ul style='margin-top: 0.5rem; margin-bottom: 1rem; padding-left: 0; list-style-type: disc;'>${listItems.join("")}</ul>`);
          listItems = [];
          inList = false;
        }
        processedLines.push(`<h2 style='margin-top: 2rem; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 600; border-bottom: 2px solid #ddd; padding-bottom: 0.5rem;'>${line.replace(/^## /, "")}</h2>`);
      } else if (line.match(/^# /)) {
        if (inList) {
          processedLines.push(`<ul style='margin-top: 0.5rem; margin-bottom: 1rem; padding-left: 0; list-style-type: disc;'>${listItems.join("")}</ul>`);
          listItems = [];
          inList = false;
        }
        processedLines.push(`<h1 style='margin-top: 2rem; margin-bottom: 1rem; font-size: 1.8rem; font-weight: 700;'>${line.replace(/^# /, "")}</h1>`);
      } else if (line.match(/^\d+\. /) || line.match(/^- /)) {
        // List item
        if (!inList) {
          inList = true;
        }
        const content = line.replace(/^(\d+\. |- )/, "");
        listItems.push(`<li style='margin-left: 1.5rem; margin-bottom: 0.5rem;'>${content}</li>`);
      } else {
        // Regular line
        if (inList) {
          processedLines.push(`<ul style='margin-top: 0.5rem; margin-bottom: 1rem; padding-left: 0; list-style-type: disc;'>${listItems.join("")}</ul>`);
          listItems = [];
          inList = false;
        }
        if (line.trim()) {
          processedLines.push(line);
        } else {
          processedLines.push("<br />");
        }
      }
    }
    
    // Close any remaining list
    if (inList) {
      processedLines.push(`<ul style='margin-top: 0.5rem; margin-bottom: 1rem; padding-left: 0; list-style-type: disc;'>${listItems.join("")}</ul>`);
    }
    
    html = processedLines.join("\n");
    
    // Process bold and italic (bold before italic to handle nested)
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    
    // Convert remaining line breaks
    html = html.replace(/\n/g, "<br />");
    
    return html;
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

          {practicePlan && (
            <section style={{ marginBottom: "3rem" }}>
              <h2 style={{ marginBottom: "1rem" }}>Practice Plan</h2>
              <div
                style={{
                  padding: "1.5rem",
                  border: "1px solid #ddd",
                  borderRadius: "8px",
                  backgroundColor: "#f9f9f9",
                  lineHeight: "1.8",
                  fontSize: "1rem",
                  fontFamily: "system-ui, -apple-system, sans-serif",
                }}
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(practicePlan),
                }}
              />
            </section>
          )}

          <section style={{ marginBottom: "3rem" }}>
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

          <section style={{ marginBottom: "3rem" }}>
            <h2 style={{ marginBottom: "1rem" }}>Identified Swing Flaws</h2>
            <textarea
              readOnly
              value={formatSwingFlaws()}
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

          {videoProcessResponse && (
            <section>
              <h2 style={{ marginBottom: "1rem" }}>Full VideoProcessResponse</h2>
              <textarea
                readOnly
                value={JSON.stringify(videoProcessResponse, null, 2)}
                style={{
                  width: "100%",
                  minHeight: "400px",
                  padding: "1rem",
                  fontSize: "0.9rem",
                  fontFamily: "monospace",
                  border: "1px solid #ddd",
                  borderRadius: "4px",
                  backgroundColor: "#f9f9f9",
                  resize: "vertical",
                  lineHeight: "1.4",
                }}
              />
            </section>
          )}
        </>
      ) : (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p>No frames available yet. Processing may still be in progress.</p>
        </div>
      )}
    </main>
  );
}

