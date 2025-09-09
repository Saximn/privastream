"use client";
import { useState, useEffect, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { FacePreview } from "@/components/enrollment/facepreview";
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

const formSchema = z.object({
  fullName: z.string().min(2, {
    message: "Full name must be at least 2 characters.",
  }),
  email: z
    .string()
    .email({
      message: "Please enter a valid email address.",
    })
    .optional()
    .or(z.literal("")),
});

export default function EnrollmentPage() {
  const router = useRouter();
  const [enrollmentStep, setEnrollmentStep] = useState<
    "intro" | "capture" | "form" | "preview" | "complete"
  >("intro");
  const [capturedPhotos, setCapturedPhotos] = useState<string[]>([]);
  
  // Face detection state
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

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      fullName: "",
      email: "",
    },
  });

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
            // Start face detection after a short delay to ensure room ID is set
            setTimeout(() => {
              if (roomId && roomId.length > 0 && enrollmentStep === "capture") {
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

    if (enrollmentStep === "capture") {
      initWebcam();
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
      }
    };
  }, [enrollmentStep, roomId]);

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
      console.log("[FRONTEND] Sending face detection request for room:", roomId);
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
        console.error("[FRONTEND] Face detection failed with status:", response.status, response.statusText);
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
      if (roomId && roomId.length > 0) {
        const frame = captureFrame();
        if (frame && !isEnrolling) {
          sendFrameForDetection(frame);
        }
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

    if (!isEnrolling && (Math.abs(canvas.width - rect.width) > 10 || Math.abs(canvas.height - rect.height) > 10)) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

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
        const progress = enrollmentFramesRef.current.length;
        setEnrollmentProgress(progress);
        console.log(`[ENROLLMENT] Progress: ${progress}/20 frames collected`);
      } else {
        console.log("[ENROLLMENT] Failed to capture frame");
      }
    };

    // Collect 20 frames over 4 seconds (200ms intervals)
    const frameInterval = setInterval(collectFrames, 200);
    console.log("[ENROLLMENT] Started collecting frames...");

    setTimeout(async () => {
      clearInterval(frameInterval);

      if (enrollmentFramesRef.current.length === 0) {
        setError("No frames captured for enrollment");
        setIsEnrolling(false);
        startLiveFaceDetection();
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
          setDetectedFaces([]);
          
          // Stop webcam stream
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
          }
          
          // Stop any detection intervals
          if (detectionIntervalRef.current) {
            clearInterval(detectionIntervalRef.current);
            detectionIntervalRef.current = null;
          }
          
          // Store captured photo for preview
          if (enrollmentFramesRef.current.length > 0) {
            setCapturedPhotos([`data:image/jpeg;base64,${enrollmentFramesRef.current[0]}`]);
          }
          // Store room ID for host page
          sessionStorage.setItem("enrolledRoomId", roomId);
          console.log("[ENROLLMENT] ✅ Face enrolled successfully for room:", roomId);
        } else {
          setError(result.message || "Enrollment failed");
          startLiveFaceDetection();
        }
      } catch (err) {
        setError("Enrollment request failed: " + (err as Error).message);
        startLiveFaceDetection();
      }

      setIsEnrolling(false);
    }, 4000); // 20 frames * 200ms = 4000ms
  };

  // Update face boxes when detections change
  useEffect(() => {
    const updateBoxes = () => {
      drawFaceBoxes();
    };
    requestAnimationFrame(updateBoxes);
  }, [detectedFaces, isEnrolling]);

  // Continuous canvas update loop for smoother bounding boxes
  useEffect(() => {
    let animationFrame: number;

    const updateCanvas = () => {
      if (!isEnrolling && enrollmentStep === "capture") {
        drawFaceBoxes();
        animationFrame = requestAnimationFrame(updateCanvas);
      }
    };

    if (videoRef.current && !isEnrolling && enrollmentStep === "capture") {
      animationFrame = requestAnimationFrame(updateCanvas);
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [detectedFaces, isEnrolling, enrollmentStep]);

  // Mock data for face preview
  const mockFaces = [
    {
      id: "2",
      name: "John Doe",
      image: "/placeholder-avatar.jpg",
      whitelisted: true,
    },
    {
      id: "3",
      name: "Unknown User",
      image: "/placeholder-avatar.jpg",
      whitelisted: false,
    },
  ];

  const handleFormSubmit = (values: z.infer<typeof formSchema>) => {
    console.log(values);
    setEnrollmentStep("preview");
  };

  const handleComplete = () => {
    setEnrollmentStep("complete");
  };

  const createRoom = () => {
    router.push("/host");
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black">
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Progress Indicator */}
          <div className="mb-8">
            <div className="flex items-center justify-center space-x-4">
              {["intro", "capture", "form", "preview", "complete"].map(
                (step, index) => (
                  <div key={step} className="flex items-center">
                    <div className="flex flex-col items-center">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                      ${
                        enrollmentStep === step
                          ? "bg-black text-white dark:bg-white dark:text-black"
                          : [
                              "intro",
                              "capture",
                              "form",
                              "preview",
                              "complete",
                            ].indexOf(enrollmentStep) > index
                          ? "bg-gray-600 text-white dark:bg-gray-400 dark:text-black"
                          : "bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400"
                      }`}
                      >
                        {[
                          "intro",
                          "capture",
                          "form",
                          "preview",
                          "complete",
                        ].indexOf(enrollmentStep) > index
                          ? "✓"
                          : index + 1}
                      </div>
                      {/* Title appears below the current active step */}
                      {enrollmentStep === step && (
                        <p className="text-sm text-gray-600 dark:text-gray-400 capitalize mt-2 whitespace-nowrap">
                          {enrollmentStep === "intro"
                            ? "Welcome"
                            : enrollmentStep === "capture"
                            ? "Face Capture"
                            : enrollmentStep === "form"
                            ? "Personal Details"
                            : enrollmentStep === "preview"
                            ? "Review & Confirm"
                            : "Completed"}
                        </p>
                      )}
                    </div>
                    {index < 4 && (
                      <div
                        className={`w-12 h-0.5 mx-2 ${
                          [
                            "intro",
                            "capture",
                            "form",
                            "preview",
                            "complete",
                          ].indexOf(enrollmentStep) > index
                            ? "bg-gray-600 dark:bg-gray-400"
                            : "bg-gray-200 dark:bg-gray-700"
                        }`}
                      />
                    )}
                  </div>
                )
              )}
            </div>
          </div>

          {/* Introduction Step */}
          {enrollmentStep === "intro" && (
            <Card className="flex flex-col h-[700px] text-center shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl text-black dark:text-white">
                  Welcome to VirtualSecure
                </CardTitle>
                <CardDescription className="text-lg max-w-2xl mx-auto text-gray-600 dark:text-gray-400">
                  Protect your privacy while streaming with intelligent face
                  recognition technology. Only whitelisted individuals appear
                  clearly in your content.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6 flex-1 flex flex-col justify-center">
                <div className="grid md:grid-cols-3 gap-4 text-left mx-auto">
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold mb-2 text-black dark:text-white">
                      Face Capture
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Secure facial recognition using advanced AI to identify
                      authorized individuals during streams.
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold mb-2 text-black dark:text-white">
                      Privacy Protection
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Non-whitelisted faces are automatically blurred in
                      real-time to protect viewer privacy.
                    </p>
                  </div>
                  <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold mb-2 text-black dark:text-white">
                      Trusted Access
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Manage your whitelist to control exactly who appears
                      clearly in your broadcasts.
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-black dark:bg-white rounded-lg border border-gray-200 dark:border-gray-700 max-w mx-auto">
                  <p className="text-sm text-white dark:text-black">
                    <span className="font-semibold">Secure & Private:</span> All
                    facial data is encrypted and processed locally. Your
                    information is never shared with third parties.
                  </p>
                </div>
              </CardContent>
              <div className="px-6 pb-6 mt-auto">
                <Button
                  onClick={() => setEnrollmentStep("capture")}
                  size="lg"
                  className="w-full mx-auto bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Start Enrollment →
                </Button>
              </div>
            </Card>
          )}

          {/* Camera Capture Step */}
          {enrollmentStep === "capture" && (
            <Card className="flex flex-col h-[700px] text-center shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center justify-center gap-2 text-black dark:text-white">
                  Face Capture
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Position your face clearly in the camera and capture a
                  high-quality photo for enrollment.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <div className="space-y-4 flex-1 flex flex-col">
                  {!enrollmentComplete ? (
                    <div
                      className="relative bg-black rounded-lg overflow-hidden flex-1"
                      style={{ minHeight: "300px" }}
                    >
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                        style={{
                          maxHeight: "400px",
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
                      {!isEnrolling && (
                        <div className="absolute top-4 left-4 bg-black bg-opacity-70 px-3 py-2 rounded">
                          <div className="flex items-center space-x-2">
                            <div
                              className={`w-2 h-2 rounded-full ${
                                detectedFaces.length > 0 ? "bg-green-500" : "bg-red-500"
                              }`}
                            />
                            <span className="text-sm text-white">
                              {detectedFaces.length > 0
                                ? `${detectedFaces.length} face(s) detected`
                                : "No faces detected"}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Enrollment Progress Overlay */}
                      {isEnrolling && (
                        <div className="absolute inset-0 bg-black bg-opacity-75 flex items-center justify-center rounded-lg z-10">
                          <div className="bg-white text-black p-6 rounded-lg text-center max-w-sm mx-4 shadow-xl">
                            <div className="text-xl font-semibold mb-4">
                              Enrolling Face...
                            </div>
                            <div className="flex justify-between text-sm mb-3">
                              <span>Collecting frames</span>
                              <span className="font-medium">{enrollmentProgress}/20</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                              <div
                                className="bg-blue-500 h-3 rounded-full transition-all duration-300 ease-out"
                                style={{ width: `${Math.max(5, (enrollmentProgress / 20) * 100)}%` }}
                              />
                            </div>
                            <div className="text-xs text-gray-600 mt-2">
                              Please keep your face still and visible
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                      <div className="text-center p-8">
                        <div className="text-6xl mb-4">✅</div>
                        <h3 className="text-xl font-semibold text-green-600 dark:text-green-400 mb-2">
                          Face Enrolled Successfully!
                        </h3>
                        <p className="text-gray-600 dark:text-gray-400">
                          Your face has been captured and enrolled for recognition.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Face Detection Info */}
                  {detectedFaces.length > 0 && !isEnrolling && !enrollmentComplete && (
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                      <h3 className="text-lg font-semibold mb-2 text-green-600 dark:text-green-400">
                        Face Detection Active
                      </h3>
                      <div className="text-sm text-gray-700 dark:text-gray-300">
                        Face detected and ready for enrollment
                      </div>
                    </div>
                  )}

                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400 px-4 py-3 rounded-lg">
                      <div className="flex items-center gap-2">
                        <span className="text-red-500">⚠️</span>
                        {error}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  variant="outline"
                  onClick={() => {
                    setEnrollmentStep("intro");
                    // Reset enrollment state if going back
                    if (enrollmentComplete) {
                      setEnrollmentComplete(false);
                      setEnrollmentProgress(0);
                      setCapturedPhotos([]);
                      setError("");
                    }
                  }}
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Back
                </Button>
                <Button
                  onClick={() => {
                    if (enrollmentComplete) {
                      setEnrollmentStep("form");
                    } else {
                      startEnrollment();
                    }
                  }}
                  disabled={detectedFaces.length === 0 && !enrollmentComplete}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200 disabled:opacity-50"
                >
                  {isEnrolling ? "Enrolling..." : enrollmentComplete ? "Continue →" : "Enroll Face"}
                </Button>
              </div>
            </Card>
          )}

          {/* Form Step */}
          {enrollmentStep === "form" && (
            <Card className="flex flex-col h-[700px] text-center shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader>
                <CardTitle className="flex items-center justify-center gap-2 text-black dark:text-white">
                  Personal Details
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Provide your information to complete the enrollment process.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <Form {...form}>
                  <form
                    onSubmit={form.handleSubmit(handleFormSubmit)}
                    className="space-y-6 flex-1 flex flex-col"
                  >
                    <div className="space-y-4 flex-1">
                      <FormField
                        control={form.control}
                        name="fullName"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel className="text-base font-medium text-black dark:text-white">
                              Full Name
                            </FormLabel>
                            <FormControl>
                              <Input
                                placeholder="Enter your full name"
                                className="w-full px-6 py-4 text-lg border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent bg-white dark:bg-black text-black dark:text-white h-14"
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name="email"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel className="text-base font-medium text-black dark:text-white">
                              Email (Optional)
                            </FormLabel>
                            <FormControl>
                              <Input
                                type="email"
                                placeholder="your.email@example.com"
                                className="w-full px-6 py-4 text-lg border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent bg-white dark:bg-black text-black dark:text-white h-14"
                                {...field}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-start gap-2">
                          <div>
                            <h4 className="font-medium text-black dark:text-white">
                              Privacy Notice
                            </h4>
                            <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                              Your facial data is encrypted and stored securely.
                              It will only be used for identification during
                              streams.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </form>
                </Form>
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEnrollmentStep("capture")}
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Back
                </Button>
                <Button
                  onClick={form.handleSubmit(handleFormSubmit)}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Review Enrollment →
                </Button>
              </div>
            </Card>
          )}

          {/* Preview Step */}
          {enrollmentStep === "preview" && (
            <Card className="flex flex-col h-[700px] shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="text-center">
                <CardTitle className="flex items-center justify-center gap-2 text-black dark:text-white">
                  Review & Confirm
                </CardTitle>
                <CardDescription className="text-gray-600 dark:text-gray-400">
                  Review your enrollment details and see how face recognition
                  will work.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col">
                <div className="space-y-6 flex-1">
                  <div className="space-y-6">
                    <div className="text-left">
                      <h3 className="font-semibold mb-3 text-black dark:text-white">
                        Your Information
                      </h3>
                      <div className="space-y-2 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 text-left">
                        <p className="text-black dark:text-white text-left">
                          <span className="font-medium">Name:</span>{" "}
                          {form.getValues("fullName")}
                        </p>
                        <p className="text-black dark:text-white text-left">
                          <span className="font-medium">Status:</span>{" "}
                          <Badge className="ml-2 bg-black text-white dark:bg-white dark:text-black">
                            Whitelisted
                          </Badge>
                        </p>
                        <p className="text-black dark:text-white text-left">
                          <span className="font-medium">Enrollment Date:</span>{" "}
                          {new Date().toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div>
                      <FacePreview
                        faces={[
                          {
                            id: "1",
                            name: form.getValues("fullName"),
                            image:
                              capturedPhotos[0] || "/placeholder-avatar.jpg",
                            whitelisted: true,
                          },
                          ...mockFaces,
                        ]}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  variant="outline"
                  onClick={() => setEnrollmentStep("form")}
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Back
                </Button>
                <Button
                  onClick={handleComplete}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Complete Enrollment
                </Button>
              </div>
            </Card>
          )}

          {/* Complete Step */}
          {enrollmentStep === "complete" && (
            <Card className="flex flex-col h-[700px] shadow-xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl mx-auto">
              <CardHeader className="pb-4">
                <div className="mx-auto mb-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                  <div className="h-8 w-8 bg-black dark:bg-white rounded-full flex items-center justify-center">
                    <span className="text-white dark:text-black text-lg">
                      ✓
                    </span>
                  </div>
                </div>
                <CardTitle className="flex items-center justify-center gap-2 text-black dark:text-white">
                  Enrollment Complete!
                </CardTitle>
                <CardDescription className="text-lg max-w-2xl mx-auto text-gray-600 dark:text-gray-400">
                  You&apos;ve been successfully added to the whitelist. You can
                  now start streaming with face recognition enabled.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-center">
                <div className="p-10 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 max-w-2xl mx-auto w-full">
                  <h3 className="font-semibold text-black dark:text-white mb-6 text-center text-xl">
                    What&apos;s Next?
                  </h3>
                  <ul className="text-base text-gray-700 dark:text-gray-300 space-y-4 text-left">
                    <li className="flex items-start gap-3">
                      <span className="text-black dark:text-white font-medium">
                        •
                      </span>
                      Your face will be recognized automatically during streams
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-black dark:text-white font-medium">
                        •
                      </span>
                      Non-whitelisted faces will be blurred for privacy
                    </li>
                    <li className="flex items-start gap-3">
                      <span className="text-black dark:text-white font-medium">
                        •
                      </span>
                      You can manage your whitelist settings anytime
                    </li>
                  </ul>
                </div>
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  onClick={() => {
                    setEnrollmentStep("intro");
                    form.reset();
                    setCapturedPhotos([]);
                  }}
                  variant="outline"
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Enroll Another Person
                </Button>
                <Button
                  onClick={createRoom}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Start Streaming →
                </Button>
              </div>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}
