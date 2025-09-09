"use client";

import { useState, useEffect, useRef } from "react";
import { SocketManager } from "@/lib/socket";
import { MediasoupClient } from "@/lib/mediasoup-client";
import { io, Socket } from "socket.io-client";

export default function Host() {
  const [roomId, setRoomId] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [viewerCount, setViewerCount] = useState(0);
  const [error, setError] = useState("");
  const [connectionState, setConnectionState] = useState("");

  // Video Filtering State
  const [isVideoFilterEnabled, setIsVideoFilterEnabled] = useState(false);
  const [videoFilterStats, setVideoFilterStats] = useState<any>(null);
  const [isPIIDetectionEnabled, setIsPIIDetectionEnabled] = useState(false);
  const [totalPIIDetected, setTotalPIIDetected] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<SocketManager | null>(null);
  const sfuSocketRef = useRef<Socket | null>(null);
  const mediasoupClientRef = useRef<MediasoupClient | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);

  useEffect(() => {
    const initializeConnections = async () => {
      try {
        console.log("[DEBUG] Starting connection initialization...");
        setError("");
        setConnectionState("connecting to backend...");

        // Signaling socket
        socketRef.current = new SocketManager();
        await socketRef.current.connect();
        console.log("[DEBUG] Connected to backend signaling socket");

        setConnectionState("creating room...");

        // Create mediasoup room
        const roomResponse = await socketRef.current.createRoom();
        console.log("[DEBUG] Room created:", roomResponse);
        setRoomId(roomResponse.roomId);

        // Check if we have enrollment data and associate it with the new room
        const enrolledRoomId = sessionStorage.getItem("enrolledRoomId");
        if (enrolledRoomId) {
          console.log(
            "[DEBUG] 🎯 Transferring enrolled face data from room:",
            enrolledRoomId,
            "to mediasoup room:",
            roomResponse.roomId
          );

          // Transfer enrollment data from enrolledRoomId to actual roomId
          try {
            const transferResponse = await fetch(
              `${process.env.NEXT_PUBLIC_VIDEO_SERVICE_URL || "http://localhost:5001"}/transfer-embedding`,
              {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  from_room_id: enrolledRoomId,
                  to_room_id: roomResponse.roomId,
                }),
              }
            );

            const transferResult = await transferResponse.json();
            if (transferResult.success) {
              console.log(
                "[DEBUG] ✅ Successfully transferred face embedding to streaming room"
              );
              sessionStorage.setItem("mediasoupRoomId", roomResponse.roomId);
            } else {
              console.warn(
                "[DEBUG] ⚠️ Failed to transfer face embedding:",
                transferResult.error
              );
            }
          } catch (error) {
            console.error(
              "[DEBUG] ❌ Error transferring face embedding:",
              error
            );
          }
        } else {
          console.log(
            "[DEBUG] ⚠️ No face enrollment found - proceeding without face recognition"
          );
        }

        // SFU socket
        setConnectionState("connecting to mediasoup...");
        const sfuUrl = roomResponse.mediasoupUrl || process.env.NEXT_PUBLIC_MEDIASOUP_URL || "http://localhost:3001";
        sfuSocketRef.current = io(sfuUrl, {
          transports: ["websocket"],
          reconnectionAttempts: 3,
        });

        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(
            () => reject(new Error("SFU connection timeout")),
            10000
          );
          sfuSocketRef.current!.on("connect", () => {
            clearTimeout(timeout);
            console.log("[DEBUG] Connected to SFU socket");
            resolve();
          });
          sfuSocketRef.current!.on("connect_error", (err) => {
            clearTimeout(timeout);
            console.error("[DEBUG] SFU connection error:", err);
            reject(err);
          });
          sfuSocketRef.current!.on("disconnect", (reason) =>
            console.warn("[DEBUG] SFU disconnected:", reason)
          );
        });

        try {
          setConnectionState("initializing mediasoup client...");
          console.log("[DEBUG] Initializing Mediasoup client...");
          mediasoupClientRef.current = new MediasoupClient(roomResponse.roomId);
          await mediasoupClientRef.current.initialize(sfuSocketRef.current!);
          console.log("[DEBUG] Mediasoup client initialized");
        } catch (err) {
          console.error("[DEBUG] Mediasoup client initialization failed:", err);
        }

        try {
          setConnectionState("creating SFU room...");
          console.log("[DEBUG] Creating SFU room via socket emit...");
          await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(
              () => reject(new Error("SFU room creation timeout")),
              5000
            );
            sfuSocketRef.current!.emit(
              "create-room",
              { roomId: roomResponse.roomId },
              (sfuResponse: any) => {
                clearTimeout(timeout);
                console.log(
                  "[DEBUG] SFU create-room callback fired",
                  sfuResponse
                );
                if (sfuResponse.success) resolve();
                else reject(new Error(sfuResponse.error));
              }
            );
          });
        } catch (err) {
          console.error("[DEBUG] SFU room creation failed:", err);
        }

        console.log("[DEBUG] All connections initialized, ready to stream");
        setConnectionState("ready");
      } catch (err) {
        console.error("[DEBUG] Initialization error:", err);
        setError(
          `Connection failed: ${
            err instanceof Error ? err.message : String(err)
          }`
        );
        setConnectionState("error");
      }
    };

    initializeConnections();
    initializeConnections();

    return () => {
      // Cleanup
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      localStreamRef.current?.getTracks().forEach((track) => track.stop());
      mediasoupClientRef.current?.stopProducing();
      socketRef.current?.disconnect();
      sfuSocketRef.current?.disconnect();
    };
  }, []);

  const startStreaming = async () => {
    try {
      setError("");
      setConnectionState("initializing");
      if (!mediasoupClientRef.current || !socketRef.current)
        throw new Error("Connections not initialized");

      console.log("[HOST] Requesting microphone and camera access...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      console.log("[HOST] 🎤 Media access granted! Stream details:", {
        id: stream.id,
        active: stream.active,
        videoTracks: stream.getVideoTracks().length,
        audioTracks: stream.getAudioTracks().length,
      });

      // Verify audio track
      const audioTrack = stream.getAudioTracks()[0];
      if (audioTrack) {
        console.log("[HOST] ✅ Audio track acquired:", {
          id: audioTrack.id,
          enabled: audioTrack.enabled,
          muted: audioTrack.muted,
          readyState: audioTrack.readyState,
        });
      } else {
        console.error("[HOST] ❌ No audio track in stream!");
      }

      localStreamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;

      console.log("[DEBUG] Creating producer transport...");
      await mediasoupClientRef.current.createProducerTransport();
      console.log(
        "[DEBUG] Producing stream with video filter:",
        isVideoFilterEnabled
      );
      await mediasoupClientRef.current.produce(stream, isVideoFilterEnabled);

      // Start frame processing for video filter
      if (isVideoFilterEnabled) {
        startFrameProcessing(stream);
      }

      // Start audio processing for redaction
      startAudioProcessing(stream);

      socketRef.current.getSocket().emit("sfu_streaming_started", { roomId });
      setIsStreaming(true);
      setConnectionState("streaming");
      console.log("[DEBUG] Streaming started successfully");
    } catch (err) {
      console.error("[DEBUG] Streaming error:", err);
      setError(
        "Failed to start streaming. Check camera/microphone permissions."
      );
      setConnectionState("error");
    }
  };

  const stopStreaming = async () => {
    try {
      // Stop frame processing
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }

      // Stop audio processing
      if (audioProcessorRef.current) {
        audioProcessorRef.current.disconnect();
        audioProcessorRef.current = null;
      }

      if (audioContextRef.current) {
        await audioContextRef.current.close();
        audioContextRef.current = null;
      }

      await mediasoupClientRef.current?.stopProducing();
      localStreamRef.current?.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;

      socketRef.current?.getSocket().emit("sfu_streaming_stopped", { roomId });
      setIsStreaming(false);
      setConnectionState("ready");
      setViewerCount(0);
      console.log("[DEBUG] Streaming stopped");
    } catch (err) {
      console.error("[DEBUG] Stop streaming error:", err);
    }
  };

  const startFrameProcessing = (stream: MediaStream) => {
    console.log("[DEBUG] Starting frame processing for video filter");

    // Create canvas for frame extraction
    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;

    // Create a video element for frame extraction
    const tempVideo = document.createElement("video");
    tempVideo.srcObject = stream;
    tempVideo.muted = true;
    tempVideo.playsInline = true;
    tempVideo.play();

    // Process frames at 4 FPS (every 250ms)
    frameIntervalRef.current = setInterval(() => {
      try {
        if (tempVideo.videoWidth > 0 && tempVideo.videoHeight > 0) {
          // Set canvas size to match video
          canvas.width = tempVideo.videoWidth;
          canvas.height = tempVideo.videoHeight;

          // Draw current video frame to canvas
          ctx.drawImage(tempVideo, 0, 0);

          // Convert to base64
          const frameData = canvas.toDataURL("image/jpeg", 0.7);
          const frameId = Date.now();

          // Send frame to MediaSoup server for processing
          if (sfuSocketRef.current) {
            sfuSocketRef.current.emit("video-frame", {
              frame: frameData,
              frameId: frameId,
              roomId: roomId,
            });

            console.log("[DEBUG] Sent video frame for processing:", frameId);
          }
        }
      } catch (error) {
        console.error("[DEBUG] Frame processing error:", error);
      }
    }, 250); // 4 FPS

    console.log("[DEBUG] Frame processing started");
  };

  const startAudioProcessing = (stream: MediaStream) => {
    console.log("[DEBUG] Starting audio processing for redaction");

    const audioTrack = stream.getAudioTracks()[0];
    if (!audioTrack) {
      console.error("[DEBUG] No audio track available for processing");
      return;
    }

    try {
      // Create audio context
      audioContextRef.current = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
      const audioContext = audioContextRef.current;

      console.log("[DEBUG] Audio context created:", {
        sampleRate: audioContext.sampleRate,
        state: audioContext.state,
      });

      // Create media stream source
      const source = audioContext.createMediaStreamSource(stream);

      // Create script processor node (deprecated but still works)
      const bufferSize = 4096; // Buffer size for processing
      const processor = audioContext.createScriptProcessor(bufferSize, 2, 2); // Stereo
      audioProcessorRef.current = processor;

      // Process audio data
      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const outputBuffer = event.outputBuffer;

        // Get left and right channel data
        const leftChannel = inputBuffer.getChannelData(0);
        const rightChannel = inputBuffer.getChannelData(1);

        // Mix stereo to mono by averaging channels
        const monoData = new Float32Array(bufferSize);
        for (let i = 0; i < bufferSize; i++) {
          monoData[i] = (leftChannel[i] + rightChannel[i]) / 2;
        }

        // Downsample from 48kHz to 16kHz (3:1 ratio)
        const downsampleRatio = 3;
        const outputSamples = Math.floor(bufferSize / downsampleRatio);
        const downsampledData = new Float32Array(outputSamples);

        for (let i = 0; i < outputSamples; i++) {
          // Simple downsampling - take every 3rd sample
          downsampledData[i] = monoData[i * downsampleRatio];
        }

        // Convert float32 to int16 PCM data (16kHz mono)
        const pcmData = new Int16Array(outputSamples);
        for (let i = 0; i < outputSamples; i++) {
          const sample = Math.max(-1, Math.min(1, downsampledData[i]));
          pcmData[i] = sample * 0x7fff;
        }

        // Send PCM data to server for processing
        if (sfuSocketRef.current) {
          sfuSocketRef.current.emit("audio-data", Array.from(pcmData));
        }

        // Copy input to output but muted to avoid feedback
        for (
          let channel = 0;
          channel < outputBuffer.numberOfChannels;
          channel++
        ) {
          const outputData = outputBuffer.getChannelData(channel);
          outputData.fill(0); // Fill with silence to prevent feedback
        }
      };

      // Connect audio processing chain
      source.connect(processor);
      processor.connect(audioContext.destination); // Connect to ensure processing happens

      console.log("[DEBUG] Audio processing pipeline connected");
    } catch (error) {
      console.error("[DEBUG] Audio processing setup error:", error);
    }
  };

  const copyRoomId = () => {
    navigator.clipboard.writeText(roomId);
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black">
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8 relative">
            <h1 className="text-4xl font-bold mb-2 text-black dark:text-white">
              VirtualSecure Stream Host
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Professional streaming with privacy protection
            </p>
            <div className="flex items-center justify-center gap-6 mt-4">
              {roomId && (
                <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg">
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Room ID</div>
                  <div className="font-mono text-lg font-bold text-black dark:text-white">
                    {roomId}
                  </div>
                </div>
              )}
              
              {/* Privacy Filter Toggle */}
              <div className="flex items-center gap-3 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Privacy Filter
                </div>
                <button
                  onClick={() => setIsVideoFilterEnabled(!isVideoFilterEnabled)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    isVideoFilterEnabled
                      ? "bg-green-500 focus:ring-green-500"
                      : "bg-gray-300 focus:ring-gray-500"
                  }`}
                  role="switch"
                  aria-checked={isVideoFilterEnabled}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-lg transition-transform ${
                      isVideoFilterEnabled ? "translate-x-6" : "translate-x-1"
                    }`}
                  />
                </button>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {isVideoFilterEnabled ? "ON" : "OFF"}
                </div>
              </div>
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400 px-4 py-3 rounded-lg mb-6">
              <div className="flex items-center gap-2">
                <span className="text-red-500">⚠️</span>
                {error}
              </div>
            </div>
          )}

          {/* Mobile-First Layout: Video on top, controls below */}
          <div className="flex flex-col lg:grid lg:grid-cols-3 gap-8">
            {/* Video Stream - First on mobile, right column on desktop */}
            <div className="order-1 lg:order-2 lg:col-span-2 flex flex-col">
              {/* Video Stream */}
              <div className="bg-black dark:bg-white border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl overflow-hidden flex-1 min-h-[300px] lg:min-h-[400px] relative">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-contain"
                  style={{ backgroundColor: "#000" }}
                  onLoadedData={async () => {
                    console.log("Video loaded and ready");
                    if (videoRef.current) {
                      try {
                        await videoRef.current.play();
                        console.log("Video playback started successfully");
                      } catch (error) {
                        console.log("Video play failed:", error);
                      }
                    }
                  }}
                  onError={(e) => {
                    console.error("Video error:", e);
                  }}
                />

                {!isStreaming && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">📹</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Ready to start streaming
                      </div>
                      <div className="text-sm mt-2 opacity-75">
                        Click "Start Stream" to begin
                      </div>
                    </div>
                  </div>
                )}

                {connectionState === "connecting" && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">🔄</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Connecting to server...
                      </div>
                      <div className="text-sm mt-2 opacity-75">Please wait</div>
                    </div>
                  </div>
                )}

                {error && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">❌</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Stream error
                      </div>
                      <div className="text-sm mt-2 opacity-75">
                        Check console for details
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Controls & Settings - Second on mobile, left column on desktop */}
            <div className="order-2 lg:order-1 lg:col-span-1 space-y-6">
              {/* Stream Controls */}
              <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-6">
                <h2 className="text-xl font-semibold text-black dark:text-white mb-4">
                  Stream Controls
                </h2>

                <div className="space-y-4">
                  <div className="flex gap-4 justify-center">
                    {!isStreaming ? (
                      <button
                        onClick={startStreaming}
                        disabled={connectionState !== "ready"}
                        className="bg-black hover:bg-gray-800 text-white dark:bg-white dark:hover:bg-gray-200 dark:text-black font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed w-full"
                      >
                        {connectionState === "ready"
                          ? "Start SFU Streaming"
                          : "Initializing..."}
                      </button>
                    ) : (
                      <button
                        onClick={stopStreaming}
                        className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg transition-colors w-full"
                      >
                        Stop Streaming
                      </button>
                    )}
                  </div>

                  <div className="text-xs text-gray-600 dark:text-gray-400 text-center space-y-1">
                    <p>
                      <strong>SFU Mode:</strong> Scalable streaming via
                      Mediasoup server
                    </p>
                    <p>Supports hundreds of concurrent viewers</p>
                    <p>
                      <strong>Privacy Protection:</strong> Real-time PII
                      detection
                    </p>
                  </div>
                </div>
              </div>

              {/* Status Dashboard */}
              <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-6">
                <h2 className="text-xl font-semibold text-black dark:text-white mb-4">
                  Status Dashboard
                </h2>

                <div className="space-y-4">
                  {/* Stream Status - Full Width */}
                  <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 p-4 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-black dark:text-white text-sm mb-1">
                          Stream Status
                        </h3>
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              isStreaming ? "bg-green-500" : "bg-red-500"
                            }`}
                          />
                          <span className="text-sm text-black dark:text-white font-medium">
                            {isStreaming ? "Live" : "Offline"}
                          </span>
                        </div>
                        {connectionState && (
                          <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                            {connectionState}
                          </div>
                        )}
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-black dark:text-white">
                          {viewerCount}
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          viewers
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* PII Detection & Room ID Row */}
                  <div className="grid grid-cols-1 gap-4">
                    <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 p-4 rounded-lg">
                      <h3 className="font-medium text-black dark:text-white text-sm mb-2">
                        PII Detection
                      </h3>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              isPIIDetectionEnabled
                                ? "bg-green-500"
                                : "bg-red-500"
                            }`}
                          />
                          <span className="text-sm text-black dark:text-white font-medium">
                            {isPIIDetectionEnabled ? "Active" : "Inactive"}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-2xl font-bold text-black dark:text-white">
                            {totalPIIDetected}
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            detected
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 p-4 rounded-lg">
                      <h3 className="font-medium text-black dark:text-white text-sm mb-2">
                        Room ID
                      </h3>
                      <div className="flex items-center gap-2">
                        <code className="bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-600 px-2 py-1 rounded text-sm font-mono text-black dark:text-white flex-1 truncate">
                          {roomId || "Generating..."}
                        </code>
                        <button
                          onClick={copyRoomId}
                          disabled={!roomId}
                          className="bg-black hover:bg-gray-800 text-white dark:bg-white dark:hover:bg-gray-200 dark:text-black text-sm px-3 py-1 rounded disabled:opacity-50 flex-shrink-0"
                        >
                          Copy
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

            </div>

          </div>
        </div>
      </main>
    </div>
  );
}
