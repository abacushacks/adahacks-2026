import { useEffect, useState } from "react";

import { VideoStage } from "./components/VideoStage";
import { useStreamingSession } from "./hooks/useStreamingSession";


function App() {
  const {
    audioGateOpen,
    connectionState,
    errorMessage,
    isRecordingVoice,
    isStreaming,
    mediaReady,
    startSession,
    stopSession,
    captureVoiceNote,
    enrollCurrentFace,
    tracking,
    videoRef,
  } = useStreamingSession();
  const [manualName, setManualName] = useState("");

  useEffect(() => {
    if (tracking?.trackingStatus === "recognized") {
      setManualName("");
    }
  }, [tracking]);

  const canEnrollCurrentFace =
    isStreaming && tracking?.trackingStatus === "unknown";

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Real-Time Dementia Support</p>
          <h1>AR Memory Aid</h1>
          <p className="lead">
            Start the session with no saved name, then record a short voice
            introduction like &quot;My name is John&quot; and later facts like
            &quot;I&apos;m your son&quot; to build the profile.
          </p>
        </div>

        <div className="control-panel">
          <div className="status-row">
            <StatusPill
              label={mediaReady ? "Camera + mic ready" : "Requesting media"}
              tone={mediaReady ? "good" : "neutral"}
            />
            <StatusPill
              label={
                isRecordingVoice
                  ? "Recording voice note"
                  : audioGateOpen
                    ? "Mouth gate open"
                    : "Mouth gate closed"
              }
              tone={isRecordingVoice || audioGateOpen ? "good" : "neutral"}
            />
            <StatusPill
              label={`Socket: ${connectionState}`}
              tone={connectionState === "connected" ? "good" : "neutral"}
            />
          </div>

          <div className="action-row">
            <button
              className="primary-button"
              disabled={!mediaReady || isStreaming}
              onClick={startSession}
            >
              Start Session
            </button>
            <button
              className="secondary-button"
              disabled={!isStreaming || isRecordingVoice || !tracking}
              onClick={captureVoiceNote}
            >
              {isRecordingVoice ? "Recording..." : "Record Voice"}
            </button>
            <button
              className="ghost-button"
              disabled={!isStreaming}
              onClick={stopSession}
            >
              Stop Session
            </button>
          </div>

          {errorMessage ? <p className="error-text">{errorMessage}</p> : null}

          <div className="manual-enroll-card">
            <p className="summary-label">Voice Flow</p>
            <p className="manual-enroll-copy">
              Keep one face visible, click `Record Voice`, then say
              &quot;My name is John&quot; to save the person. After that, record
              another clip and say facts like &quot;I&apos;m your son&quot; or
              &quot;My birthday is June 4&quot;.
            </p>
          </div>

          <div className="manual-enroll-card">
            <p className="summary-label">Manual Fallback</p>
            <p className="manual-enroll-copy">
              Use this only if the voice clip does not name the person correctly.
            </p>
            <div className="manual-enroll-row">
              <input
                className="manual-enroll-input"
                type="text"
                value={manualName}
                onChange={(event) => setManualName(event.target.value)}
                placeholder="Enter subject name"
                disabled={!canEnrollCurrentFace}
              />
              <button
                className="secondary-button"
                disabled={!canEnrollCurrentFace || manualName.trim().length === 0}
                onClick={() => enrollCurrentFace(manualName)}
              >
                Enroll Current Face
              </button>
            </div>
          </div>

          <div className="summary-card">
            <p className="summary-label">Current subject</p>
            <p className="summary-name">
              {tracking?.trackingStatus === "recognized" && tracking.metadata
                ? tracking.metadata.name
                : "No recognized subject"}
            </p>
            <p className="summary-detail">
              {tracking?.trackingStatus === "recognized" &&
              tracking.metadata?.relationship
                ? tracking.metadata.relationship
                : "Start the session and record a voice clip to identify the subject."}
            </p>
          </div>
        </div>
      </section>

      <VideoStage tracking={tracking} videoRef={videoRef} />
    </main>
  );
}


type StatusPillProps = {
  label: string;
  tone: "good" | "neutral";
};


function StatusPill({ label, tone }: StatusPillProps) {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}


export default App;
