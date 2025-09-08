"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

interface FaceDetection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
}

interface EnrollmentResponse {
  success: boolean;
  faces_detected?: FaceDetection[];
  enrollment_complete?: boolean;
  message?: string;
  metadata?: {
    enrollment_time: string;
    frames_processed: number;
    valid_frames: number;
    embeddings_count: number;
  };
}

export default function CreatorEnrollment() {
  const router = useRouter();
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollmentComplete, setEnrollmentComplete] = useState(false);
  const [error, setError] = useState("");
  const [detectedFaces, setDetectedFaces] = useState<FaceDetection[]>([]);
  const [enrollmentProgress, setEnrollmentProgress] = useState(0);
  const [roomId, setRoomId] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const enrollmentFramesRef = useRef<string[]>([]);

  // Generate room ID on component mount
  useEffect(() => {
    const generateRoomId = () => {
      const id = Math.random().toString(36).substring(2, 15);
      console.log("[ENROLLMENT] Generated room ID:", id);
      setRoomId(id);
      return id;
    };
    generateRoomId();
  }, []);

  // Start face detection when room ID is available and video is ready
  useEffect(() => {
    console.log("[ENROLLMENT] Room ID useEffect triggered, roomId:", roomId);
    console.log(
      "[ENROLLMENT] Video ref:",
      !!videoRef.current,
      "srcObject:",
      !!videoRef.current?.srcObject
    );

    if (roomId && videoRef.current && videoRef.current.srcObject) {
      console.log("[ENROLLMENT] Room ID available, starting face detection");
      startLiveFaceDetection();
    } else {
      console.log("[ENROLLMENT] Conditions not met for starting detection");
    }
  }, [roomId]);

  // Start face detection when video stream becomes available
  useEffect(() => {
    const checkVideoReady = () => {
      console.log("[ENROLLMENT] Checking if video is ready...");
      console.log(
        "[ENROLLMENT] roomId:",
        roomId,
        "video ref:",
        !!videoRef.current,
        "srcObject:",
        !!videoRef.current?.srcObject
      );

      if (roomId && videoRef.current && videoRef.current.srcObject) {
        console.log(
          "[ENROLLMENT] Video stream is ready, starting face detection"
        );
        startLiveFaceDetection();
        return true;
      }
      return false;
    };

    // Check immediately
    if (!checkVideoReady()) {
      // If not ready, set up a polling interval to check periodically
      const interval = setInterval(() => {
        if (checkVideoReady()) {
          clearInterval(interval);
        }
      }, 100); // Check every 100ms

      // Clean up interval after 10 seconds
      setTimeout(() => {
        clearInterval(interval);
        console.log("[ENROLLMENT] Timeout waiting for video stream");
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [roomId]);

  // Initialize webcam
  useEffect(() => {
    const initWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: "user",
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            // Try to start face detection after a short delay to ensure room ID is set
            setTimeout(() => {
              if (roomId && roomId.length > 0) {
                startLiveFaceDetection();
              }
            }, 100);
          };
        }
        streamRef.current = stream;
      } catch (err) {
        setError("Failed to access webcam: " + (err as Error).message);
      }
    };

    initWebcam();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, []);

  const captureFrame = (): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    return canvas.toDataURL("image/jpeg", 0.8);
  };

  const sendFrameForDetection = async (frameData: string) => {
    try {
      console.log(
        "[FRONTEND] Sending face detection request for room:",
        roomId
      );
      console.log("[FRONTEND] Room ID length:", roomId?.length || 0);
      console.log("[FRONTEND] Frame data length:", frameData?.length || 0);
      const response = await fetch("http://localhost:5001/face-detection", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          frame_data: frameData.split(",")[1], // Remove data:image/jpeg;base64, prefix
          room_id: roomId,
          detect_only: true,
        }),
      });

      if (response.ok) {
        const result: EnrollmentResponse = await response.json();
        console.log("[FRONTEND] Face detection response:", result);
        setDetectedFaces(result.faces_detected || []);
      } else {
        console.error(
          "[FRONTEND] Face detection failed with status:",
          response.status,
          response.statusText
        );
        const errorText = await response.text();
        console.error("[FRONTEND] Error response:", errorText);
      }
    } catch (err) {
      console.error("[FRONTEND] Face detection error:", err);
    }
  };

  const startLiveFaceDetection = () => {
    console.log("[ENROLLMENT] startLiveFaceDetection called, roomId:", roomId);

    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    detectionIntervalRef.current = setInterval(() => {
      // Only send if we have a valid room ID
      if (roomId && roomId.length > 0) {
        console.log(
          "[ENROLLMENT] Attempting to capture frame, roomId:",
          roomId
        );
        const frame = captureFrame();
        if (frame && !isEnrolling) {
          console.log("[ENROLLMENT] Frame captured, sending for detection");
          sendFrameForDetection(frame);
        } else if (!frame) {
          console.log("[ENROLLMENT] Failed to capture frame");
        } else if (isEnrolling) {
          console.log("[ENROLLMENT] Skipping detection - currently enrolling");
        }
      } else {
        console.log(
          "[ENROLLMENT] Waiting for room ID to be generated...",
          "roomId:",
          roomId
        );
      }
    }, 500); // Detect faces every 500ms
  };

  const drawFaceBoxes = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) return;

    // Set canvas size to match video display size
    const rect = video.getBoundingClientRect();

    // Only resize canvas if dimensions changed significantly (prevent during enrollment to avoid zoom)
    if (
      !isEnrolling &&
      (Math.abs(canvas.width - rect.width) > 10 ||
        Math.abs(canvas.height - rect.height) > 10)
    ) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    // If canvas hasn't been sized yet, size it initially
    if (canvas.width === 0 || canvas.height === 0) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    // Always clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Only draw if we have faces
    if (detectedFaces.length > 0) {
      const scaleX = rect.width / video.videoWidth;
      const scaleY = rect.height / video.videoHeight;

      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 2;

      detectedFaces.forEach((face) => {
        const [x1, y1, x2, y2] = face.bbox;
        const x = x1 * scaleX;
        const y = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;

        // Draw bounding box only
        ctx.strokeRect(x, y, width, height);
      });
    }
  };

  const startEnrollment = async () => {
    if (detectedFaces.length === 0) {
      setError("No faces detected. Please ensure your face is visible.");
      return;
    }

    setIsEnrolling(true);
    setError("");
    setEnrollmentProgress(0);
    enrollmentFramesRef.current = [];

    // Stop live detection during enrollment
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    // Collect frames for enrollment
    const collectFrames = () => {
      const frame = captureFrame();
      if (frame) {
        enrollmentFramesRef.current.push(frame.split(",")[1]); // Remove data URL prefix
        setEnrollmentProgress(enrollmentFramesRef.current.length);
      }
    };

    // Collect 20 frames over 4 seconds (200ms intervals)
    const frameInterval = setInterval(collectFrames, 200);

    setTimeout(async () => {
      clearInterval(frameInterval);

      if (enrollmentFramesRef.current.length === 0) {
        setError("No frames captured for enrollment");
        setIsEnrolling(false);
        startLiveFaceDetection(); // Resume live detection
        return;
      }

      try {
        const response = await fetch("http://localhost:5001/face-enrollment", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            frames: enrollmentFramesRef.current,
            room_id: roomId,
          }),
        });

        const result: EnrollmentResponse = await response.json();

        if (result.success && result.enrollment_complete) {
          setEnrollmentComplete(true);
          setDetectedFaces([]); // Clear live detection
          // Store room ID for host page
          sessionStorage.setItem("enrolledRoomId", roomId);
          console.log(
            "[ENROLLMENT] ✅ Face enrolled successfully for room:",
            roomId
          );
        } else {
          setError(result.message || "Enrollment failed");
          startLiveFaceDetection(); // Resume live detection on failure
        }
      } catch (err) {
        setError("Enrollment request failed: " + (err as Error).message);
        startLiveFaceDetection(); // Resume live detection on error
      }

      setIsEnrolling(false);
    }, 4000); // 20 frames * 200ms = 4000ms
  };

  const proceedToHosting = () => {
    router.push("/host");
  };

  // Update face boxes when detections change
  useEffect(() => {
    // Use requestAnimationFrame for smoother rendering
    const updateBoxes = () => {
      drawFaceBoxes();
    };

    // Always update when faces change
    requestAnimationFrame(updateBoxes);
  }, [detectedFaces, isEnrolling]);

  // Continuous canvas update loop for smoother bounding boxes (disabled during enrollment)
  useEffect(() => {
    let animationFrame: number;

    const updateCanvas = () => {
      if (!isEnrolling) {
        drawFaceBoxes();
        animationFrame = requestAnimationFrame(updateCanvas);
      }
    };

    if (videoRef.current && !isEnrolling) {
      animationFrame = requestAnimationFrame(updateCanvas);
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [detectedFaces, isEnrolling]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">Creator Face Enrollment</h1>
          <p className="text-gray-400">
            Position your face in the camera view and click "Enroll Face" to
            proceed to streaming
          </p>
          {roomId && (
            <p className="text-sm text-gray-500 mt-2">Room ID: {roomId}</p>
          )}
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Video Preview */}
          <div className="space-y-4">
            <div
              className="relative bg-black rounded-lg overflow-hidden"
              style={{ minHeight: "300px" }}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-auto object-cover"
                style={{
                  maxHeight: "480px",
                  minHeight: "300px",
                  transform: "scaleX(-1)", // Mirror effect
                  position: "relative",
                  zIndex: 1,
                }}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 pointer-events-none"
                style={{
                  transform: "scaleX(-1)", // Mirror effect to match video
                  width: "100%",
                  height: "100%",
                  zIndex: 2,
                }}
              />

              {/* Face Detection Status */}
              <div className="absolute top-4 left-4 bg-black bg-opacity-70 px-3 py-2 rounded">
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      detectedFaces.length > 0 ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span className="text-sm">
                    {detectedFaces.length > 0
                      ? `${detectedFaces.length} face(s) detected`
                      : "No faces detected"}
                  </span>
                </div>
              </div>

              {/* Enrollment Progress Overlay */}
              {isEnrolling && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                  <div className="bg-white text-black p-4 rounded-lg text-center max-w-xs mx-4 shadow-lg">
                    <div className="text-base font-semibold mb-3">
                      Enrolling Face...
                    </div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Collecting frames</span>
                      <span>{enrollmentProgress}/20</span>
                    </div>
                    <div className="w-full max-w-48 bg-gray-200 rounded-full h-2 mx-auto">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(enrollmentProgress / 20) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Face Detection Info */}
            {detectedFaces.length > 0 && !isEnrolling && (
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2 text-green-400">
                  Face Detection Active
                </h3>
                <div className="text-sm text-gray-300">
                  Primary face detected and ready for enrollment
                </div>
              </div>
            )}
          </div>

          {/* Enrollment Panel */}
          <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-lg">
              <h2 className="text-2xl font-semibold mb-4">Face Enrollment</h2>

              {!enrollmentComplete ? (
                <>
                  <div className="space-y-4 mb-6">
                    <p className="text-gray-300">
                      Make sure your face is clearly visible and well-lit before
                      enrolling.
                    </p>

                    {isEnrolling && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Collecting frames...</span>
                          <span>{enrollmentProgress}/20</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all"
                            style={{
                              width: `${(enrollmentProgress / 20) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}

                    {error && (
                      <div className="bg-red-900 border border-red-500 p-3 rounded text-red-100">
                        {error}
                      </div>
                    )}
                  </div>

                  <button
                    onClick={startEnrollment}
                    disabled={isEnrolling || detectedFaces.length === 0}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold transition-colors"
                  >
                    {isEnrolling ? "Enrolling..." : "Enroll Face"}
                  </button>
                </>
              ) : (
                <div className="text-center space-y-4">
                  <div className="text-green-400 text-6xl mb-4">✓</div>
                  <h3 className="text-xl font-semibold text-green-400">
                    Enrollment Complete!
                  </h3>
                  <p className="text-gray-300">
                    Your face has been successfully enrolled for this streaming
                    session.
                  </p>

                  <button
                    onClick={proceedToHosting}
                    className="w-full bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg font-semibold transition-colors"
                  >
                    Start Streaming
                  </button>
                </div>
              )}
            </div>

            {/* Instructions */}
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="font-semibold mb-2">Instructions:</h3>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• Face the camera directly</li>
                <li>• Ensure good lighting</li>
                <li>• Stay still during enrollment</li>
                <li>• Only one face should be visible</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
