"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [ballShape, setBallShape] = useState<string>("");
  const [contact, setContact] = useState<string>("");
  const [description, setDescription] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null); // Clear previous result
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    if (ballShape) {
      formData.append("ball_shape", ballShape);
    }
    if (contact) {
      formData.append("contact", contact);
    }
    if (description) {
      formData.append("description", description);
    }

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
      const response = await fetch(`${apiBase}/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        // Store the full VideoProcessResponse in localStorage
        localStorage.setItem(`videoResponse_${data.video_id}`, JSON.stringify(data));
        setResult("Upload successful! Redirecting to results...");
        // Redirect to results page after a brief delay
        setTimeout(() => {
          router.push(`/results/${data.video_id}`);
        }, 1000);
      } else {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        setResult(`Upload failed: ${errorData.detail || response.statusText}`);
        setUploading(false);
      }
    } catch (error) {
      setResult(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
      setUploading(false);
    }
  };

  return (
    <main style={{ padding: "2rem", maxWidth: "600px", margin: "0 auto" }}>
      <h1>Upload Golf Swing Video</h1>
      <p style={{ marginTop: "0.5rem", color: "#666" }}>
        Upload a video of your golf swing to analyze key frames and metrics.
      </p>
      <div style={{ marginTop: "2rem" }}>
        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500" }}>
            Video File
          </label>
          <input
            type="file"
            accept="video/mp4,video/mov"
            onChange={handleFileChange}
            disabled={uploading}
            style={{
              padding: "0.5rem",
              fontSize: "1rem",
              width: "100%",
              border: "1px solid #ddd",
              borderRadius: "4px",
              cursor: uploading ? "not-allowed" : "pointer",
            }}
          />
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500" }}>
            Ball Shape
          </label>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            {["left", "straight", "right"].map((shape) => (
              <label
                key={shape}
                style={{
                  display: "flex",
                  alignItems: "center",
                  cursor: uploading ? "not-allowed" : "pointer",
                  opacity: uploading ? 0.6 : 1,
                }}
              >
                <input
                  type="radio"
                  name="ballShape"
                  value={shape}
                  checked={ballShape === shape}
                  onChange={(e) => setBallShape(e.target.value)}
                  disabled={uploading}
                  style={{ marginRight: "0.5rem" }}
                />
                <span style={{ textTransform: "capitalize" }}>{shape}</span>
              </label>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500" }}>
            Contact
          </label>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            {["fat", "thin", "normal", "inconsistent"].map((contactType) => (
              <label
                key={contactType}
                style={{
                  display: "flex",
                  alignItems: "center",
                  cursor: uploading ? "not-allowed" : "pointer",
                  opacity: uploading ? 0.6 : 1,
                }}
              >
                <input
                  type="radio"
                  name="contact"
                  value={contactType}
                  checked={contact === contactType}
                  onChange={(e) => setContact(e.target.value)}
                  disabled={uploading}
                  style={{ marginRight: "0.5rem" }}
                />
                <span style={{ textTransform: "capitalize" }}>{contactType}</span>
              </label>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: "500" }}>
            Description of Issues
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={uploading}
            placeholder="Describe any issues you're experiencing with your swing..."
            rows={6}
            style={{
              padding: "0.75rem",
              fontSize: "1rem",
              width: "100%",
              border: "1px solid #ddd",
              borderRadius: "4px",
              fontFamily: "inherit",
              resize: "vertical",
              cursor: uploading ? "not-allowed" : "text",
            }}
          />
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          style={{
            padding: "0.75rem 1.5rem",
            fontSize: "1rem",
            backgroundColor: uploading ? "#ccc" : "#0070f3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: uploading ? "not-allowed" : "pointer",
            fontWeight: "500",
            transition: "background-color 0.2s",
          }}
        >
          {uploading ? "Uploading..." : "Upload Video"}
        </button>
        {result && (
          <p
            style={{
              marginTop: "1rem",
              padding: "0.75rem",
              borderRadius: "4px",
              backgroundColor: result.includes("successful") ? "#d4edda" : "#f8d7da",
              color: result.includes("successful") ? "#155724" : "#721c24",
              border: `1px solid ${result.includes("successful") ? "#c3e6cb" : "#f5c6cb"}`,
            }}
          >
            {result}
          </p>
        )}
      </div>
    </main>
  );
}

