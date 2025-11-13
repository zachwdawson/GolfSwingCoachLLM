import Link from "next/link";

export default function Home() {
  return (
    <main style={{ padding: "2rem", maxWidth: "800px", margin: "0 auto", textAlign: "center" }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>Golf Swing Coach</h1>
      <p style={{ fontSize: "1.2rem", color: "#666", marginBottom: "2rem" }}>
        Analyze your golf swing with AI-powered frame extraction and biomechanical metrics
      </p>
      <div style={{ marginTop: "3rem" }}>
        <Link
          href="/upload"
          style={{
            display: "inline-block",
            padding: "1rem 2rem",
            fontSize: "1.1rem",
            backgroundColor: "#0070f3",
            color: "white",
            textDecoration: "none",
            borderRadius: "6px",
            fontWeight: "500",
            transition: "background-color 0.2s",
          }}
        >
          Upload Your Swing Video
        </Link>
      </div>
      <div style={{ marginTop: "3rem", textAlign: "left", maxWidth: "600px", margin: "3rem auto 0" }}>
        <h2 style={{ fontSize: "1.5rem", marginBottom: "1rem" }}>Features</h2>
        <ul style={{ listStyle: "none", padding: 0 }}>
          <li style={{ padding: "0.5rem 0", fontSize: "1.1rem" }}>
            ✓ Automatic key frame extraction (Address, Top, Mid-downswing, Impact, Finish)
          </li>
          <li style={{ padding: "0.5rem 0", fontSize: "1.1rem" }}>
            ✓ Pose estimation with keypoint annotations
          </li>
          <li style={{ padding: "0.5rem 0", fontSize: "1.1rem" }}>
            ✓ Biomechanical metrics analysis
          </li>
          <li style={{ padding: "0.5rem 0", fontSize: "1.1rem" }}>
            ✓ Visual feedback on swing positions
          </li>
        </ul>
      </div>
    </main>
  );
}

