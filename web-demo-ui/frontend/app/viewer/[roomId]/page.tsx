"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import { io, Socket } from "socket.io-client";
import * as mediasoupClient from "mediasoup-client";
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

export default function Viewer() {
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
  const [streamAvailable, setStreamAvailable] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const sfuSocketRef = useRef<Socket | null>(null);
  const deviceRef = useRef<mediasoupClient.Device | null>(null);
  const consumerTransportRef = useRef<mediasoupClient.types.Transport | null>(
    null
  );
  const consumersRef = useRef<Map<string, mediasoupClient.types.Consumer>>(
    new Map()
  );
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processedFramesRef = useRef<Map<string, any>>(new Map());
  const canvasStreamRef = useRef<MediaStream | null>(null);
  const isStreamInitialized = useRef<boolean>(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioBufferSourceRef = useRef<AudioBufferSourceNode | null>(null);

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
		path:"/mediasoup/socket.io",
          transports: ["websocket"],
          reconnectionAttempts: 3
        });

        await new Promise<void>((resolve) => {
          sfuSocketRef.current!.on("connect", () => {
            console.log("[VIEWER] Connected to SFU server");
            resolve();
          });
        });

        // Initialize MediaSoup device
        console.log("[VIEWER] Initializing MediaSoup device...");
        deviceRef.current = new mediasoupClient.Device();

        // Get router RTP capabilities
        const rtpCapabilities = await new Promise<any>((resolve, reject) => {
          sfuSocketRef.current!.emit(
            "getRouterRtpCapabilities",
            { roomId },
            (response: any) => {
              console.log("[VIEWER] RTP capabilities response:", response);
              if (response.error) {
                reject(new Error(response.error));
              } else {
                resolve(response.rtpCapabilities);
              }
            }
          );
        });

        await deviceRef.current.load({
          routerRtpCapabilities: rtpCapabilities,
        });
        console.log("[VIEWER] Device loaded");

        // Join room as viewer
        console.log("[VIEWER] Joining room as viewer...");
        const joinResponse = await new Promise<any>((resolve, reject) => {
          sfuSocketRef.current!.emit(
            "join-room",
            { roomId },
            (response: any) => {
              console.log("[VIEWER] Join room response:", response);
              if (response.success) {
                resolve(response);
              } else {
                reject(new Error(response.error || "Failed to join room"));
              }
            }
          );
        });

        // Create consumer transport
        console.log("[VIEWER] Creating consumer transport...");
        const transportParams = await new Promise<any>((resolve, reject) => {
          sfuSocketRef.current!.emit(
            "createConsumerTransport",
            { roomId },
            (response: any) => {
              console.log("[VIEWER] Consumer transport response:", response);
              if (response.error) {
                reject(new Error(response.error));
              } else {
                resolve(response);
              }
            }
          );
        });

        consumerTransportRef.current =
          deviceRef.current.createRecvTransport(transportParams);

        // Handle transport connection
        consumerTransportRef.current.on(
          "connect",
          ({ dtlsParameters }, callback, errback) => {
            console.log("[VIEWER] Consumer transport connecting...");
            sfuSocketRef.current!.emit(
              "connectConsumerTransport",
              {
                roomId,
                dtlsParameters,
              },
              (response: any) => {
                if (response.error) {
                  errback(new Error(response.error));
                } else {
                  callback();
                }
              }
            );
          }
        );

        console.log("[VIEWER] Consumer transport created");

        // Set up event handlers
        setupEventHandlers();
        setupProcessedFrameHandlers();
        setupAudioHandlers();

        // Request existing producers (but skip audio producers since we use processed audio)
        console.log("[VIEWER] Requesting existing producers...");
        sfuSocketRef.current!.emit(
          "getProducers",
          { roomId },
          (response: any) => {
            console.log("[VIEWER] Existing producers:", response);
            if (response.producers) {
              response.producers.forEach((producer: any) => {
                // Only consume video producers, skip audio (we use processed audio instead)
                if (producer.kind === "video") {
                  consumeProducer(producer.id, producer.kind);
                } else {
                  console.log(
                    "[VIEWER] Skipping audio producer - using processed audio instead"
                  );
                }
              });
            }
          }
        );

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

      // Handle new producers (but skip audio producers since we use processed audio)
      sfuSocketRef.current.on("new-producer", (data: any) => {
        console.log("[VIEWER] New producer:", data);
        // Only consume video producers, skip audio (we use processed audio instead)
        if (data.kind === "video") {
          consumeProducer(data.producerId, data.kind);
        } else {
          console.log(
            "[VIEWER] Skipping new audio producer - using processed audio instead"
          );
        }
      });

      // Handle producer closed
      sfuSocketRef.current.on("producer-closed", (data: any) => {
        console.log("[VIEWER] Producer closed:", data);
        const consumer = consumersRef.current.get(data.producerId);
        if (consumer) {
          consumer.close();
          consumersRef.current.delete(data.producerId);
        }
        updateVideoElement();
      });

      // Handle host disconnected
      sfuSocketRef.current.on("host-disconnected", () => {
        console.log("[VIEWER] Host disconnected");
        setError("Host disconnected");
        setStats((prev) => ({ ...prev, hostStreaming: false }));
        setStreamAvailable(false);

        // Clear all consumers
        consumersRef.current.forEach((consumer) => consumer.close());
        consumersRef.current.clear();
        updateVideoElement();
      });
    };

    const setupProcessedFrameHandlers = () => {
      if (!sfuSocketRef.current) return;

      // Handle processed video frames with bounding box data
      sfuSocketRef.current.on("processed-video-frame", (data: any) => {
        console.log("[VIEWER] Received video frame:", {
          frameId: data.frameId,
          boundingBoxCount: data.boundingBoxCount,
          wasDetectionFrame: data.wasDetectionFrame,
        });

        // Store processed frame with metadata
        processedFramesRef.current.set(data.frameId, {
          frame: data.frame,
          boundingBoxCount: data.boundingBoxCount || 0,
          wasDetectionFrame: data.wasDetectionFrame || false,
          timestamp: data.timestamp,
        });

        // Display the frame (with client-side blur if needed)
        displayProcessedFrame(data.frame, data.boundingBoxCount);

        // Update stats and stream availability
        setStats((prev) => ({
          ...prev,
          receivingVideo: true,
          hostStreaming: true,
        }));
        setStreamAvailable(true);

        // Clean up old frames (keep last 30)
        if (processedFramesRef.current.size > 30) {
          const frames = Array.from(processedFramesRef.current.keys());
          const oldestFrame = frames[0];
          processedFramesRef.current.delete(oldestFrame);
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

    const displayProcessedFrame = (
      frameData: string,
      boundingBoxCount: number = 0
    ) => {
      if (!videoRef.current) return;

      try {
        // Create canvas if needed
        if (!canvasRef.current) {
          canvasRef.current = document.createElement("canvas");
        }

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d")!;

        // Create image from base64 data
        const img = new Image();
        img.onload = () => {
          // Set canvas size (only if changed)
          if (canvas.width !== img.width || canvas.height !== img.height) {
            canvas.width = img.width;
            canvas.height = img.height;

            // Initialize stream only once when canvas size is set
            if (!isStreamInitialized.current) {
              canvasStreamRef.current = canvas.captureStream(30);
              videoRef.current!.srcObject = canvasStreamRef.current;
              isStreamInitialized.current = true;
              console.log(
                "[VIEWER] Initialized canvas stream:",
                `${img.width}x${img.height}`
              );
            }
          }

          // Clear and draw new frame
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);

          // Add privacy protection indicator if bounding boxes were detected
          if (boundingBoxCount > 0) {
            ctx.fillStyle = "rgba(0, 255, 0, 0.8)";
            ctx.fillRect(10, 10, 200, 30);
            ctx.fillStyle = "black";
            ctx.font = "14px Arial";
            ctx.fillText(`🛡️ Privacy: ${boundingBoxCount} regions`, 15, 30);
          }

          // Canvas stream automatically updates when canvas content changes
          console.log("[VIEWER] Updated frame:", {
            size: `${img.width}x${img.height}`,
            privacyRegions: boundingBoxCount,
          });
        };
        img.src = frameData;
      } catch (error) {
        console.error("[VIEWER] Error displaying processed frame:", error);
      }
    };

    const consumeProducer = async (
      producerId: string,
      kind: "audio" | "video"
    ) => {
      try {
        if (!consumerTransportRef.current || !deviceRef.current) {
          console.error("[VIEWER] Transport or device not ready");
          return;
        }

        console.log(`[VIEWER] Consuming ${kind} producer:`, producerId);

        // Get consumer parameters from server
        const consumerParams = await new Promise<any>((resolve, reject) => {
          sfuSocketRef.current!.emit(
            "consume",
            {
              roomId,
              producerId,
              rtpCapabilities: deviceRef.current!.rtpCapabilities,
            },
            (response: any) => {
              console.log(`[VIEWER] Consume ${kind} response:`, response);
              if (response.error) {
                reject(new Error(response.error));
              } else {
                resolve(response);
              }
            }
          );
        });

        // Create consumer
        const consumer = await consumerTransportRef.current.consume({
          id: consumerParams.id,
          producerId: consumerParams.producerId,
          kind: consumerParams.kind,
          rtpParameters: consumerParams.rtpParameters,
        });

        console.log(`[VIEWER] Created ${kind} consumer:`, consumer.id);
        consumersRef.current.set(producerId, consumer);

        // Resume consumer
        sfuSocketRef.current!.emit(
          "resumeConsumer",
          {
            roomId,
            consumerId: consumer.id,
          },
          (response: any) => {
            console.log(`[VIEWER] Resume ${kind} consumer response:`, response);
          }
        );

        // Update stats
        setStats((prev) => ({
          ...prev,
          hostStreaming: true,
          receivingVideo: kind === "video" || prev.receivingVideo,
          receivingAudio: kind === "audio" || prev.receivingAudio,
        }));
        
        // Set stream as available when we have video
        if (kind === "video") {
          setStreamAvailable(true);
        }

        // Update video element
        updateVideoElement();
      } catch (err) {
        console.error(`[VIEWER] Failed to consume ${kind} producer:`, err);
      }
    };

    const updateVideoElement = () => {
      if (!videoRef.current) return;

      const videoConsumers = Array.from(consumersRef.current.values()).filter(
        (consumer) => consumer.kind === "video"
      );
      const audioConsumers = Array.from(consumersRef.current.values()).filter(
        (consumer) => consumer.kind === "audio"
      );

      console.log(
        "[VIEWER] Updating video element - Video consumers:",
        videoConsumers.length,
        "Audio consumers:",
        audioConsumers.length
      );

      if (videoConsumers.length > 0 || audioConsumers.length > 0) {
        const tracks: MediaStreamTrack[] = [];

        videoConsumers.forEach((consumer) => tracks.push(consumer.track));
        audioConsumers.forEach((consumer) => tracks.push(consumer.track));

        const stream = new MediaStream(tracks);
        videoRef.current.srcObject = stream;

        console.log(
          "[VIEWER] Set stream with tracks:",
          tracks.map((t) => `${t.kind}:${t.id}`)
        );
      } else {
        videoRef.current.srcObject = null;
        console.log("[VIEWER] Cleared video element");
      }
    };

    initializeViewer();

    return () => {
      console.log("[VIEWER] Cleanup...");

      // Close all consumers
      consumersRef.current.forEach((consumer) => consumer.close());
      consumersRef.current.clear();

      // Close transport
      if (consumerTransportRef.current) {
        consumerTransportRef.current.close();
      }

      // Disconnect socket
      if (sfuSocketRef.current) {
        sfuSocketRef.current.disconnect();
      }

      // Clear processed frames and streams
      processedFramesRef.current.clear();

      // Clean up canvas stream
      if (canvasStreamRef.current) {
        canvasStreamRef.current.getTracks().forEach((track) => track.stop());
        canvasStreamRef.current = null;
      }
      isStreamInitialized.current = false;

      // Clean up audio context
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }

      if (audioBufferSourceRef.current) {
        audioBufferSourceRef.current.stop();
        audioBufferSourceRef.current = null;
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
          ? "Watching Live Stream"
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
      <div className="min-h-screen bg-white dark:bg-black">
        <main className="container mx-auto px-4 py-8">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold mb-2 text-black dark:text-white">
                Loading VirtualSecure...
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Initializing secure connection
              </p>
            </div>
            <div className="bg-black dark:bg-white border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl overflow-hidden">
              <video
                autoPlay
                muted={true}
                playsInline
                controls
                className="w-full h-auto object-contain"
                style={{ backgroundColor: "#000", minHeight: "400px" }}
              />
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white dark:bg-black">
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-2 text-black dark:text-white">
              VirtualSecure Stream Viewer
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Secure viewing with privacy protection
            </p>
            {roomId && (
              <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg inline-block">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Viewing Room</div>
                <div className="font-mono text-lg font-bold text-black dark:text-white">
                  {roomId}
                </div>
              </div>
            )}
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
                  muted={true}
                  playsInline
                  controls
                  className="w-full h-full object-contain"
                  style={{ backgroundColor: "#000" }}
                  onLoadedData={async () => {
                    console.log("Video loaded and ready to play");
                    // Always start muted and playing
                    if (videoRef.current) {
                      try {
                        videoRef.current.muted = true;
                        await videoRef.current.play();
                        console.log("Video autoplay successful (muted)");
                        // Mark stream as available when video actually starts playing
                        setStreamAvailable(true);
                      } catch (error) {
                        console.log("Autoplay failed:", error);
                      }
                    }
                  }}
                  onPlaying={() => {
                    console.log("Video is playing");
                    setStreamAvailable(true);
                  }}
                  onError={(e) => {
                    console.error("Video error:", e);
                  }}
                />

                {!streamAvailable && connectionState === "connected" && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">⏳</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Waiting for host to start streaming...
                      </div>
                      <div className="text-sm mt-2 opacity-75">
                        Connected to SFU server
                      </div>
                    </div>
                  </div>
                )}

                {connectionState === "connecting" && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">🔄</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Connecting to SFU server...
                      </div>
                      <div className="text-sm mt-2 opacity-75">Please wait</div>
                    </div>
                  </div>
                )}

                {(connectionState === "error" ||
                  connectionState === "disconnected") && (
                  <div className="absolute inset-0 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <div className="text-center">
                      <div className="text-4xl lg:text-6xl mb-4">❌</div>
                      <div className="text-gray-400 dark:text-gray-600 text-lg">
                        Connection failed
                      </div>
                      <div className="text-sm mt-2 opacity-75">
                        Please refresh the page
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Audio Controls - Show right under video on mobile */}
              {isMuted && streamAvailable && (
                <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 mt-4 lg:hidden">
                  <button
                    onClick={handleUnmute}
                    className="bg-black hover:bg-gray-800 text-white dark:bg-white dark:hover:bg-gray-200 dark:text-black font-bold py-3 px-6 rounded-lg transition-colors w-full flex items-center justify-center gap-2"
                  >
                    <span className="text-xl">🔊</span>
                    Unmute Audio
                  </button>
                  <div className="text-xs text-gray-600 dark:text-gray-400 text-center mt-2">
                    Stream starts muted for autoplay compatibility
                  </div>
                </div>
              )}
            </div>

            {/* Status & Controls - Second on mobile, left column on desktop */}
            <div className="order-2 lg:order-1 lg:col-span-1 space-y-6">
              {/* Connection Status */}
              <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-6">
                <h2 className="text-xl font-semibold text-black dark:text-white mb-4">
                  Connection Status
                </h2>

                <div className="space-y-4">
                  <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 p-4 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-black dark:text-white text-sm mb-1">
                          Stream Status
                        </h3>
                        <div className="flex items-center gap-2">
                          <div
                            className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`}
                          />
                          <span className="text-sm text-black dark:text-white font-medium">
                            {connectionState === "connected" && streamAvailable
                              ? "Live"
                              : connectionState === "connected"
                              ? "Waiting"
                              : connectionState === "connecting"
                              ? "Connecting"
                              : "Offline"}
                          </span>
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {getConnectionStatusText()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>


              {/* Audio Controls */}
              {isMuted && streamAvailable && (
                <div className="bg-white dark:bg-black border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-6">
                  <h2 className="text-xl font-semibold text-black dark:text-white mb-4">
                    Audio Controls
                  </h2>

                  <button
                    onClick={handleUnmute}
                    className="bg-black hover:bg-gray-800 text-white dark:bg-white dark:hover:bg-gray-200 dark:text-black font-bold py-3 px-6 rounded-lg transition-colors w-full flex items-center justify-center gap-2"
                  >
                    <span className="text-xl">�</span>
                    Unmute Audio
                  </button>

                  <div className="text-xs text-gray-600 dark:text-gray-400 text-center mt-2">
                    Stream starts muted for autoplay compatibility
                  </div>
                </div>
              )}
            </div>

          </div>

          {/* Footer Information */}
          <div className="text-center mt-8">
            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <p>
                <strong>SFU Mode:</strong> Optimized streaming via Mediasoup
                server
              </p>
              <p>Low latency, high quality viewing experience</p>
              <p>
                <strong>Privacy Protected:</strong> Sensitive content
                automatically redacted
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
