"use client";

import { useState } from "react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
      const response = await fetch(`${apiBase}/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(`Upload successful! Video ID: ${data.video_id}`);
      } else {
        setResult(`Upload failed: ${response.statusText}`);
      }
    } catch (error) {
      setResult(`Error: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <main style={{ padding: "2rem", maxWidth: "600px", margin: "0 auto" }}>
      <h1>Upload Golf Swing Video</h1>
      <div style={{ marginTop: "2rem" }}>
        <input
          type="file"
          accept="video/mp4,video/mov"
          onChange={handleFileChange}
          style={{ marginBottom: "1rem" }}
        />
        <br />
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          style={{
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            cursor: uploading ? "not-allowed" : "pointer",
          }}
        >
          {uploading ? "Uploading..." : "Upload"}
        </button>
        {result && (
          <p style={{ marginTop: "1rem", color: result.includes("successful") ? "green" : "red" }}>
            {result}
          </p>
        )}
      </div>
    </main>
  );
}

