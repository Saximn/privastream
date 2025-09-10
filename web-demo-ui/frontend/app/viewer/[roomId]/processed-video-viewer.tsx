"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import { io, Socket } from "socket.io-client";
import API_CONFIG from "@/lib/config";

interface ViewerStats {
  connected: boolean;
  roomExists: boolean;
  hostStreaming: boolean;
  receivingVideo: boolean;
  receivingAudio: boolean;
  bufferHealth: "good" | "low" | "critical";
  latency: number;
}

export default function ProcessedVideoViewer() {
  const params = useParams();
  const roomId = params.roomId as string;

  const [connectionState, setConnectionState] = useState<
    "connecting" | "connected" | "error" | "disconnected"
  >("connecting");
  const [error, setError] = useState("");
  const [stats, setStats] = useState<ViewerStats>({
    connected: false,
    roomExists: false,
    hostStreaming: false,
    receivingVideo: false,
    receivingAudio: false,
    bufferHealth: "good",
    latency: 0,
  });
  const [isMuted, setIsMuted] = useState(true);
  const [mounted, setMounted] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const sfuSocketRef = useRef<Socket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasStreamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const initializeViewer = async () => {
      try {
        setError("");
        setConnectionState("connecting");
        console.log("[VIEWER] Initializing for room:", roomId);

        // Connect to SFU server
        console.log("[VIEWER] Connecting to SFU server...");
        sfuSocketRef.current = io(API_CONFIG.SFU_URL, {
          transports: ["websocket"],
          reconnectionAttempts: 3
        }
        );

        await new Promise<void>((resolve) => {
          sfuSocketRef.current!.on("connect", () => {
            console.log("[VIEWER] Connected to SFU server");
            resolve();
          });
        });

        // Set up event handlers
        setupEventHandlers();
        setupAudioHandlers();
        setupProcessedVideoHandlers();

        // Join the room to get streaming status
        console.log("[VIEWER] Joining room as viewer...");
        sfuSocketRef.current!.emit("join-room", { roomId }, (response: any) => {
          console.log("[VIEWER] Join room response:", response);
          if (response.success) {
            // Check if host is already streaming
            if (response.hostStreaming) {
              console.log("[VIEWER] 🎉 Host is already streaming!");
              setStats((prev) => ({ ...prev, hostStreaming: true }));
            }
          }
        });

        setConnectionState("connected");
        setStats((prev) => ({ ...prev, connected: true, roomExists: true }));
      } catch (err) {
        console.error("[VIEWER] Initialization error:", err);
        setError(
          `Failed to connect: ${
            err instanceof Error ? err.message : "Unknown error"
          }`
        );
        setConnectionState("error");
      }
    };

    const setupEventHandlers = () => {
      if (!sfuSocketRef.current) return;

      // Handle host started streaming
      sfuSocketRef.current.on("host-streaming-started", (data: any) => {
        console.log("[VIEWER] 🎉 Host started streaming:", data);
        setStats((prev) => ({ ...prev, hostStreaming: true }));
      });

      // Handle host disconnected
      sfuSocketRef.current.on("host-disconnected", () => {
        console.log("[VIEWER] Host disconnected");
        setError("Host disconnected");
        setStats((prev) => ({
          ...prev,
          hostStreaming: false,
          receivingVideo: false,
          receivingAudio: false,
        }));

        // Clear video display
        if (canvasStreamRef.current) {
          canvasStreamRef.current.getTracks().forEach((track) => track.stop());
          canvasStreamRef.current = null;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
      });
    };

    const setupAudioHandlers = () => {
      if (!sfuSocketRef.current) return;

      // Initialize Audio Context for processed audio playback
      try {
        audioContextRef.current = new (window.AudioContext ||
          (window as any).webkitAudioContext)();
        console.log(
          "[VIEWER] Audio context initialized for processed audio playback"
        );
      } catch (error) {
        console.error("[VIEWER] Failed to initialize audio context:", error);
      }

      // Handle processed audio from server
      sfuSocketRef.current.on("processed-audio", (data: any) => {
        console.log("[VIEWER] Received processed audio:", {
          dataSize: data.audioData.length,
          piiCount: data.metadata?.pii_count || 0,
          timestamp: data.timestamp,
        });

        // Play processed audio
        playProcessedAudio(data.audioData);

        // Update stats
        setStats((prev) => ({
          ...prev,
          receivingAudio: true,
          hostStreaming: true,
        }));
      });
    };

    const setupProcessedVideoHandlers = () => {
      if (!sfuSocketRef.current) return;

      console.log("[VIEWER] 🎥 Setting up processed video frame handlers");

      // Create canvas for displaying processed video frames
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
        canvasRef.current.width = 1280;
        canvasRef.current.height = 720;
      }

      // Handle processed video frames from server
      sfuSocketRef.current.on("processed-video-frame", (data: any) => {
        console.log("[VIEWER] 🎬 Received processed video frame:", {
          boundingBoxes: data.boundingBoxes?.length || 0,
          timestamp: data.timestamp,
          frameSize: data.frameData?.length || 0,
          processingMode: data.processingMode,
        });

        displayProcessedVideoFrame(data.frameData, data.boundingBoxes || []);

        // Update stats
        setStats((prev) => ({
          ...prev,
          receivingVideo: true,
          hostStreaming: true,
        }));
      });
    };

    const displayProcessedVideoFrame = (
      frameDataUrl: string,
      boundingBoxes: number[][]
    ) => {
      if (!canvasRef.current || !videoRef.current) return;

      try {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d")!;

        // Create image from processed frame data
        const img = new Image();
        img.onload = () => {
          // Clear and draw processed frame
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

          // Add privacy protection indicator if bounding boxes were detected
          if (boundingBoxes.length > 0) {
            ctx.fillStyle = "rgba(0, 255, 0, 0.8)";
            ctx.fillRect(10, 10, 280, 30);
            ctx.fillStyle = "black";
            ctx.font = "16px Arial";
            ctx.fillText(
              `🛡️ Privacy Protected: ${boundingBoxes.length} regions`,
              15,
              30
            );
          }

          // Add processing indicator
          ctx.fillStyle = "rgba(0, 0, 255, 0.8)";
          ctx.fillRect(10, canvas.height - 40, 200, 30);
          ctx.fillStyle = "white";
          ctx.font = "14px Arial";
          ctx.fillText("🔄 Real-time Processing", 15, canvas.height - 20);

          // Initialize canvas stream if not already done
          if (!canvasStreamRef.current) {
            canvasStreamRef.current = canvas.captureStream(30);
            videoRef.current!.srcObject = canvasStreamRef.current;
            console.log(
              "[VIEWER] ✅ Canvas stream initialized for processed video"
            );
          }
        };
        img.src = frameDataUrl;
      } catch (error) {
        console.error(
          "[VIEWER] ❌ Error displaying processed video frame:",
          error
        );
      }
    };

    const playProcessedAudio = async (audioData: number[]) => {
      if (!audioContextRef.current) return;

      try {
        const audioContext = audioContextRef.current;

        // Convert int16 array back to float32 audio buffer
        const pcmData = new Int16Array(audioData);
        const audioBuffer = audioContext.createBuffer(1, pcmData.length, 16000); // Mono, 16kHz

        // Convert to float32 mono
        const monoChannel = audioBuffer.getChannelData(0);

        for (let i = 0; i < audioBuffer.length; i++) {
          monoChannel[i] = pcmData[i] / 32767.0;
        }

        // Create and play audio buffer source
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();

        console.log(
          "[VIEWER] Playing processed audio chunk:",
          audioBuffer.duration,
          "seconds"
        );
      } catch (error) {
        console.error("[VIEWER] Error playing processed audio:", error);
      }
    };

    initializeViewer();

    return () => {
      console.log("[VIEWER] Cleanup...");

      // Clean up canvas stream
      if (canvasStreamRef.current) {
        canvasStreamRef.current.getTracks().forEach((track) => track.stop());
        canvasStreamRef.current = null;
      }

      // Disconnect socket
      if (sfuSocketRef.current) {
        sfuSocketRef.current.disconnect();
      }

      // Clean up audio context
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, [roomId, mounted]);

  const handleUnmute = () => {
    if (videoRef.current) {
      videoRef.current.muted = false;
      setIsMuted(false);
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionState) {
      case "connected":
        return "bg-green-500";
      case "connecting":
        return "bg-yellow-500";
      case "error":
      case "disconnected":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionState) {
      case "connected":
        return stats.hostStreaming
          ? "Watching Processed Stream"
          : "Connected - Waiting for Stream";
      case "connecting":
        return "Connecting...";
      case "error":
        return "Connection Error";
      case "disconnected":
        return "Disconnected";
      default:
        return "Unknown";
    }
  };

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-center mb-6 text-gray-800">
            🔄 Processed Video Stream Viewer
          </h1>

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}

          {/* Status Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Room</h3>
              <code className="bg-gray-200 px-2 py-1 rounded text-sm font-mono">
                {roomId}
              </code>
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Connection</h3>
              <div className="text-sm">
                <div
                  className={`inline-block w-2 h-2 rounded-full mr-2 ${getConnectionStatusColor()}`}
                />
                {getConnectionStatusText()}
              </div>
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Video</h3>
              <div className="text-sm">
                <div
                  className={`inline-block w-2 h-2 rounded-full mr-2 ${
                    stats.receivingVideo ? "bg-green-500" : "bg-gray-400"
                  }`}
                />
                {stats.receivingVideo ? "Processing" : "Waiting"}
              </div>
            </div>

            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-700 mb-2">Audio</h3>
              <div className="text-sm">
                <div
                  className={`inline-block w-2 h-2 rounded-full mr-2 ${
                    stats.receivingAudio ? "bg-green-500" : "bg-gray-400"
                  }`}
                />
                {stats.receivingAudio ? "Processing" : "Waiting"}
              </div>
            </div>
          </div>

          <div className="text-sm text-gray-600 text-center">
            <p>
              <strong>Flow:</strong> Host → Real-time Processing → Privacy
              Protection → You
            </p>
            <p>✅ Real-time video with face blur • ✅ Audio redaction</p>
          </div>
        </div>

        {/* Video Player */}
        <div className="bg-black rounded-lg overflow-hidden relative">
          <video
            ref={videoRef}
            autoPlay
            muted={isMuted}
            playsInline
            controls
            className="w-full h-auto object-contain"
            style={{ backgroundColor: "#000", minHeight: "400px" }}
            onLoadedData={() => {
              console.log("[VIEWER] Video loaded");
              if (videoRef.current) {
                videoRef.current.play().catch((err) => {
                  console.log("[VIEWER] Autoplay failed:", err);
                });
              }
            }}
            onError={(e) => {
              console.error("[VIEWER] Video error:", e);
            }}
          />

          {/* Unmute Button */}
          {isMuted && stats.receivingAudio && (
            <div
              className="absolute bottom-4 right-4 bg-black bg-opacity-75 text-white p-3 rounded-lg cursor-pointer z-10 hover:bg-opacity-90 transition-all duration-200"
              onClick={handleUnmute}
            >
              <div className="flex items-center space-x-2">
                <div className="text-2xl">🔊</div>
                <div className="text-sm font-medium">Unmute</div>
              </div>
            </div>
          )}

          {/* Waiting States */}
          {!stats.hostStreaming && connectionState === "connected" && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">⏳</div>
                <div>Waiting for host to start streaming...</div>
                <div className="text-sm mt-2 opacity-75">Room: {roomId}</div>
              </div>
            </div>
          )}

          {connectionState === "connecting" && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4 animate-spin">🔄</div>
                <div>Connecting to stream...</div>
                <div className="text-sm mt-2 opacity-75">Please wait</div>
              </div>
            </div>
          )}

          {(connectionState === "error" ||
            connectionState === "disconnected") && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-6xl mb-4">❌</div>
                <div>Connection failed</div>
                <div className="text-sm mt-2 opacity-75">
                  {error || "Please refresh the page"}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
