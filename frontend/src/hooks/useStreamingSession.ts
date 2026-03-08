import { useCallback, useEffect, useRef, useState } from "react";

import type {
  ConnectionState,
  FaceBoxPayload,
  ServerMessage,
  TrackingUpdateMessage,
} from "../types/protocol";
import { float32ToBase64Pcm, mergeFloat32Chunks } from "../utils/audio";


const DETECTION_INTERVAL_MS = 120;
const VOICE_CAPTURE_MS = 3000;
const FACE_API_SCRIPT_URL =
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/dist/face-api.js";
const FACE_API_MODEL_URL =
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model";
const SPEECH_GATE_HOLD_MS = 1500;
const LIP_OPENING_THRESHOLD = 0.05;
const WS_URL =
  import.meta.env.VITE_WS_URL ??
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8000/ws/session/`;


let faceApiPromise: Promise<any> | null = null;


function loadFaceApi(): Promise<any> {
  if ((window as Window & { faceapi?: any }).faceapi) {
    return Promise.resolve((window as Window & { faceapi?: any }).faceapi);
  }

  if (faceApiPromise) {
    return faceApiPromise;
  }

  faceApiPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${FACE_API_SCRIPT_URL}"]`,
    );

    const loadModels = () => {
      const faceapi = (window as Window & { faceapi?: any }).faceapi;
      if (!faceapi) {
        reject(new Error("face-api.js failed to load."));
        return;
      }

      Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODEL_URL),
      ])
        .then(() => resolve(faceapi))
        .catch(reject);
    };

    if (existing) {
      if ((window as Window & { faceapi?: any }).faceapi) {
        loadModels();
      } else {
        existing.addEventListener("load", loadModels, { once: true });
        existing.addEventListener(
          "error",
          () => reject(new Error("Failed to load face-api.js.")),
          { once: true },
        );
      }
      return;
    }

    const script = document.createElement("script");
    script.src = FACE_API_SCRIPT_URL;
    script.async = true;
    script.onload = loadModels;
    script.onerror = () => reject(new Error("Failed to load face-api.js."));
    document.head.appendChild(script);
  });

  return faceApiPromise;
}


function selectPrimaryDetection(detections: any[]): any | null {
  if (detections.length === 0) {
    return null;
  }

  return detections.reduce((largest, current) => {
    const largestArea =
      largest.detection.box.width * largest.detection.box.height;
    const currentArea =
      current.detection.box.width * current.detection.box.height;
    return currentArea > largestArea ? current : largest;
  });
}


function computeMouthActivity(detection: any): number {
  const mouth = detection.landmarks?.getMouth?.();
  const box = detection.detection?.box;
  if (!mouth || mouth.length <= 18 || !box?.height) {
    return 0;
  }

  const upperLip = mouth[14];
  const lowerLip = mouth[18];
  return Math.abs(upperLip.y - lowerLip.y) / box.height;
}


function toFacePayload(detection: any, video: HTMLVideoElement): FaceBoxPayload {
  const box = detection.detection.box;
  return {
    x: Math.round(box.x),
    y: Math.round(box.y),
    width: Math.round(box.width),
    height: Math.round(box.height),
    frameWidth: video.videoWidth,
    frameHeight: video.videoHeight,
  };
}


export function useStreamingSession() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioNodeRef = useRef<AudioWorkletNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const faceApiRef = useRef<any>(null);
  const detectionIntervalRef = useRef<number | null>(null);
  const detectionBusyRef = useRef(false);
  const facePresentRef = useRef(false);
  const voiceCaptureChunksRef = useRef<Float32Array[]>([]);
  const isRecordingVoiceRef = useRef(false);
  const isStreamingRef = useRef(false);
  const lastSpeechTimeRef = useRef(0);

  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [isRecordingVoice, setIsRecordingVoice] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [mediaReady, setMediaReady] = useState(false);
  const [tracking, setTracking] = useState<TrackingUpdateMessage | null>(null);
  const [audioGateOpen, setAudioGateOpen] = useState(false);

  const stopDetectionLoop = useCallback(() => {
    if (detectionIntervalRef.current !== null) {
      window.clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    detectionBusyRef.current = false;
    facePresentRef.current = false;
    lastSpeechTimeRef.current = 0;
    setAudioGateOpen(false);
  }, []);

  const stopSession = useCallback(() => {
    stopDetectionLoop();

    const socket = socketRef.current;
    socketRef.current = null;

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "session.stop" }));
      socket.close();
    } else if (socket) {
      socket.close();
    }

    isStreamingRef.current = false;
    setIsStreaming(false);
    setConnectionState("idle");
    setTracking(null);
  }, [stopDetectionLoop]);

  const sendAudioChunk = useCallback((chunk: Float32Array) => {
    const socket = socketRef.current;
    const audioContext = audioContextRef.current;

    if (
      !socket ||
      !audioContext ||
      socket.readyState !== WebSocket.OPEN ||
      chunk.length === 0
    ) {
      return;
    }

    socket.send(
      JSON.stringify({
        type: "media.audio",
        audio: float32ToBase64Pcm(chunk),
        sampleRate: audioContext.sampleRate,
        timestamp: Date.now(),
      }),
    );
  }, []);

  const handleServerMessage = useCallback((message: ServerMessage) => {
    if (message.type === "tracking_update") {
      setTracking(message);
      return;
    }

    if (message.type === "tracking_lost") {
      setTracking(null);
      return;
    }

    if (message.type === "error") {
      setErrorMessage(message.message);
      return;
    }

    if (message.type === "profile_learned") {
      setErrorMessage("");
    }
  }, []);

  const startDetectionLoop = useCallback(() => {
    stopDetectionLoop();

    detectionIntervalRef.current = window.setInterval(async () => {
      if (detectionBusyRef.current) {
        return;
      }

      const faceapi = faceApiRef.current;
      const video = videoRef.current;
      const socket = socketRef.current;

      if (
        !faceapi ||
        !video ||
        !socket ||
        socket.readyState !== WebSocket.OPEN ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        return;
      }

      detectionBusyRef.current = true;

      try {
        const detections = await faceapi
          .detectAllFaces(
            video,
            new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }),
          )
          .withFaceLandmarks()
          .withFaceDescriptors();

        const primary = selectPrimaryDetection(detections);

        if (!primary) {
          if (facePresentRef.current) {
            socket.send(JSON.stringify({ type: "media.face_lost" }));
            facePresentRef.current = false;
          }
          setAudioGateOpen(false);
          return;
        }

        const mouthActivity = computeMouthActivity(primary);
        if (mouthActivity >= LIP_OPENING_THRESHOLD) {
          lastSpeechTimeRef.current = Date.now();
        }

        setAudioGateOpen(
          Date.now() - lastSpeechTimeRef.current < SPEECH_GATE_HOLD_MS,
        );

        socket.send(
          JSON.stringify({
            type: "media.face",
            descriptor: Array.from(primary.descriptor),
            face: toFacePayload(primary, video),
            mouthActivity,
            timestamp: Date.now(),
          }),
        );
        facePresentRef.current = true;
      } catch (error) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "Face detection failed in the browser.",
        );
      } finally {
        detectionBusyRef.current = false;
      }
    }, DETECTION_INTERVAL_MS);
  }, [stopDetectionLoop]);

  const startSession = useCallback(async () => {
    if (isStreamingRef.current) {
      return;
    }

    const audioContext = audioContextRef.current;
    if (audioContext) {
      await audioContext.resume();
    }

    setErrorMessage("");
    setConnectionState("connecting");

    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.addEventListener("open", () => {
      setConnectionState("connected");
      isStreamingRef.current = true;
      setIsStreaming(true);

      socket.send(JSON.stringify({ type: "session.start" }));
      startDetectionLoop();
    });

    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data) as ServerMessage;
      handleServerMessage(message);
    });

    socket.addEventListener("close", () => {
      stopDetectionLoop();
      isStreamingRef.current = false;
      setIsStreaming(false);
      setConnectionState("idle");
    });

    socket.addEventListener("error", () => {
      setErrorMessage("WebSocket connection failed.");
      stopSession();
    });
  }, [handleServerMessage, startDetectionLoop, stopDetectionLoop, stopSession]);

  const enrollCurrentFace = useCallback((name: string) => {
    const socket = socketRef.current;
    const trimmed = name.trim();

    if (!trimmed) {
      setErrorMessage("Enter a name before enrolling the current face.");
      return;
    }

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setErrorMessage("Start the session before enrolling a face.");
      return;
    }

    if (tracking?.trackingStatus !== "unknown") {
      setErrorMessage("Enrollment requires a visible unrecognized face.");
      return;
    }

    setErrorMessage("");
    socket.send(
      JSON.stringify({
        type: "profile.enroll",
        name: trimmed,
      }),
    );
  }, [tracking]);

  const captureVoiceNote = useCallback(async () => {
    const audioContext = audioContextRef.current;
    if (!audioContext) {
      setErrorMessage("Audio pipeline is not ready yet.");
      return;
    }

    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setErrorMessage("Start the session before recording voice.");
      return;
    }

    if (!tracking) {
      setErrorMessage("Keep one face visible while recording voice.");
      return;
    }

    await audioContext.resume();

    voiceCaptureChunksRef.current = [];
    isRecordingVoiceRef.current = true;
    setIsRecordingVoice(true);
    setErrorMessage("");

    await new Promise((resolve) => {
      window.setTimeout(resolve, VOICE_CAPTURE_MS);
    });

    isRecordingVoiceRef.current = false;
    setIsRecordingVoice(false);

    const merged = mergeFloat32Chunks(voiceCaptureChunksRef.current);
    if (merged.length === 0) {
      setErrorMessage("No voice sample was captured.");
      return;
    }

    socket.send(
      JSON.stringify({
        type: "profile.voice_note",
        audio: float32ToBase64Pcm(merged),
        sampleRate: audioContext.sampleRate,
      }),
    );
  }, [tracking]);

  useEffect(() => {
    let isMounted = true;

    const initializeMedia = async () => {
      try {
        const [faceapi, stream] = await Promise.all([
          loadFaceApi(),
          navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: "user",
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
          }),
        ]);

        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        faceApiRef.current = faceapi;
        mediaStreamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => undefined);
        }

        const audioContext = new AudioContext();
        await audioContext.audioWorklet.addModule("/audio-processor.js");

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(
          audioContext,
          "pcm-stream-processor",
        );
        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;

        source.connect(workletNode);
        workletNode.connect(silentGain);
        silentGain.connect(audioContext.destination);

        workletNode.port.onmessage = (
          event: MessageEvent<Float32Array>,
        ) => {
          const chunk = new Float32Array(event.data);

          if (isRecordingVoiceRef.current) {
            voiceCaptureChunksRef.current.push(chunk);
          }

          if (isStreamingRef.current) {
            sendAudioChunk(chunk);
          }
        };

        audioContextRef.current = audioContext;
        audioNodeRef.current = workletNode;
        silentGainRef.current = silentGain;
        setMediaReady(true);
      } catch (error) {
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "Unable to access camera, microphone, or face-api models.",
        );
      }
    };

    initializeMedia();

    return () => {
      isMounted = false;
      stopSession();

      mediaStreamRef.current?.getTracks().forEach((track: MediaStreamTrack) => {
        track.stop();
      });
      audioNodeRef.current?.disconnect();
      silentGainRef.current?.disconnect();
      audioContextRef.current?.close().catch(() => undefined);
      mediaStreamRef.current = null;
      audioContextRef.current = null;
      audioNodeRef.current = null;
      silentGainRef.current = null;
      faceApiRef.current = null;
    };
  }, [sendAudioChunk, stopSession]);

  return {
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
  };
}
