"use client";

import React, { useRef, useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { User } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import Image from "next/image";

interface CameraCaptureProps {
  onPhotosChange?: (photos: string[]) => void;
}

export function CameraCapture({ onPhotosChange }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCameraStarted, setIsCameraStarted] = useState(false);
  const [capturedPhotos, setCapturedPhotos] = useState<string[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [, setShowCaptureSuccess] = useState(false);

  // Callback ref to handle video element setup
  const videoCallbackRef = (element: HTMLVideoElement | null) => {
    if (element && mediaStream) {
      element.srcObject = mediaStream;
      element.addEventListener("loadedmetadata", () => {
        element.play().catch(console.error);
        setIsVideoPlaying(true);
      });
      element.onerror = (error) => {
        console.error("Video error:", error);
      };
    }
  };

  const handleStartCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      });
      setMediaStream(stream);
      setIsCameraStarted(true);
      setShowPreview(true);
    } catch (err) {
      setError(
        "Error accessing webcam. Please ensure camera permissions are granted."
      );
      console.error("Error accessing webcam", err);
    }
  };
  //https://strongbase.hashnode.dev/how-to-use-a-webcam-in-the-nextjs-application
  useEffect(() => {
    if (videoRef.current && mediaStream && isCameraStarted) {
      const video = videoRef.current;
      video.srcObject = mediaStream;
      video.addEventListener("loadedmetadata", () => {
        video.play().catch(console.error);
      });
    }
  }, [videoRef, mediaStream, isCameraStarted]);

  // Additional effect to ensure video stream is set when dialog opens
  useEffect(() => {
    if (showPreview && videoRef.current && mediaStream) {
      const video = videoRef.current;
      video.srcObject = mediaStream;
      video.addEventListener("loadedmetadata", () => {
        video.play().catch(console.error);
        setIsVideoPlaying(true);
      });
    }
  }, [showPreview, mediaStream]);

  //Capture Photo function
  const handleCapturePhoto = () => {
    //exit function if there is no video element
    if (!videoRef.current) return;
    //creates canvas in memory and capture the current camera frame
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    //https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D
    const ctx = canvas.getContext("2d");
    if (ctx) {
      //copies video frame to canvas
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      //convert canvas image to base64 and stores it as text string in memory
      const dataUrl = canvas.toDataURL("image/png");
      const newPhotos = [...capturedPhotos, dataUrl];
      setCapturedPhotos(newPhotos);

      // Notify parent component of the change
      onPhotosChange?.(newPhotos);

      // Show success message briefly
      setShowCaptureSuccess(true);
      setTimeout(() => setShowCaptureSuccess(false), 2000);
    }
  };

  const handleClosePreview = () => {
    setShowPreview(false);
    setIsVideoPlaying(false);
    setShowCaptureSuccess(false);
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => {
        track.stop();
      });
      setMediaStream(null);
      setIsCameraStarted(false);
    }
  };

  const handleRetakePhoto = () => {
    setCapturedPhotos([]);
    // Notify parent component of the change
    onPhotosChange?.([]);
  };

  const handleDeletePhoto = (indexToDelete: number) => {
    const newPhotos = capturedPhotos.filter(
      (_, index) => index !== indexToDelete
    );
    setCapturedPhotos(newPhotos);
    // Notify parent component of the change
    onPhotosChange?.(newPhotos);
  };

  useEffect(() => {
    return () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, [mediaStream]);

  return (
    <>
      <Card className="mb-4 p-6 bg-white dark:bg-black border-2 border-dashed border-gray-300 dark:border-gray-600">
        {error && (
          <div className="text-red-500 dark:text-red-400 text-center mb-4 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
            {error}
          </div>
        )}

        <div className="text-center">
          {capturedPhotos.length === 0 && (
            <>
              <div className="mx-auto mb-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
                <User />
              </div>

              <h3 className="text-xl font-semibold mb-2 text-black dark:text-white">
                Face Capture
              </h3>

              <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">
                {!isCameraStarted
                  ? "Click the button below to start your camera and capture your face for enrollment."
                  : "Camera is ready. Position your face in the preview and capture your photo."}
              </p>
            </>
          )}

          {/* Show captured photos if any */}
          {capturedPhotos.length > 0 && (
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3 text-black dark:text-white">
                Captured Photos ({capturedPhotos.length}/4):
              </h4>
              <div className="grid grid-cols-4 gap-3 max-w-md mx-auto">
                {[0, 1, 2, 3].map((index) => (
                  <div
                    key={index}
                    className="aspect-square rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 flex items-center justify-center relative"
                  >
                    {capturedPhotos[index] ? (
                      <>
                        <Image
                          src={capturedPhotos[index]}
                          alt={`Captured photo ${index + 1}`}
                          className="w-full h-full object-cover rounded-lg border-2 border-black dark:border-white"
                          width={100}
                          height={100}
                        />
                        {/* Delete button */}
                        <button
                          onClick={() => handleDeletePhoto(index)}
                          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-xs font-bold shadow-lg transition-colors duration-200 z-10"
                          aria-label={`Delete photo ${index + 1}`}
                        >
                          ✕
                        </button>
                      </>
                    ) : (
                      <div className="text-gray-400 dark:text-gray-500 text-xs text-center">
                        Photo {index + 1}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button
              className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
              size="lg"
              onClick={handleStartCamera}
            >
              {capturedPhotos.length >= 4
                ? "Retake Photos"
                : capturedPhotos.length > 0
                ? "Take More Photos"
                : "Start Camera"}
            </Button>
            {capturedPhotos.length > 0 && (
              <Button
                variant="outline"
                className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                onClick={handleRetakePhoto}
              >
                Clear All Photos
              </Button>
            )}
          </div>
        </div>
      </Card>

      {/* Camera Preview Dialog */}
      <Dialog
        open={showPreview && isCameraStarted}
        onOpenChange={(open) => {
          if (!open) {
            handleClosePreview();
          }
        }}
      >
        <DialogContent className="max-w-4xl w-full max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>Camera Preview</DialogTitle>
          </DialogHeader>

          {/* Video Preview */}
          <div className="text-center">
            {!isVideoPlaying && mediaStream && (
              <div className="mb-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                <p className="text-yellow-700 dark:text-yellow-300 text-sm">
                  Setting up video stream...
                </p>
              </div>
            )}
            <div className="relative inline-block">
              <video
                ref={(el) => {
                  videoRef.current = el;
                  videoCallbackRef(el);
                }}
                autoPlay
                playsInline
                muted
                width="640"
                height="480"
                className="rounded-lg shadow-lg border-4 border-black dark:border-white block"
                style={{
                  transform: "scaleX(-1)", // Mirror effect for better UX
                  maxWidth: "100%",
                  height: "auto",
                }}
              />
            </div>

            <p className="text-sm text-gray-600 dark:text-gray-400 mt-4 mb-6">
              Make sure your face is clearly visible and well-lit before
              capturing
            </p>

            {/* Captured Photos Grid in Dialog */}
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3 text-black dark:text-white">
                Captured Photos ({capturedPhotos.length}/4):
              </h4>
              <div className="grid grid-cols-4 gap-3 max-w-md mx-auto">
                {[0, 1, 2, 3].map((index) => (
                  <div
                    key={index}
                    className="aspect-square rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 flex items-center justify-center relative"
                  >
                    {capturedPhotos[index] ? (
                      <>
                        <Image
                          src={capturedPhotos[index]}
                          alt={`Captured photo ${index + 1}`}
                          className="w-full h-full object-cover rounded-lg border-2 border-black dark:border-white"
                          width={100}
                          height={100}
                        />
                        {/* Delete button */}
                        <button
                          onClick={() => handleDeletePhoto(index)}
                          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-xs font-bold shadow-lg transition-colors duration-200 z-10"
                          aria-label={`Delete photo ${index + 1}`}
                        >
                          ✕
                        </button>
                      </>
                    ) : (
                      <div className="text-gray-400 dark:text-gray-500 text-xs text-center">
                        Photo {index + 1}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 justify-center">
              <Button
                className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                size="lg"
                onClick={handleCapturePhoto}
                disabled={capturedPhotos.length >= 4}
              >
                {capturedPhotos.length >= 4
                  ? "Max Photos Reached"
                  : "Capture Photo"}
              </Button>
              {capturedPhotos.length > 0 && (
                <Button
                  variant="outline"
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                  onClick={handleRetakePhoto}
                >
                  Clear All
                </Button>
              )}
              <Button
                variant="outline"
                className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                onClick={handleClosePreview}
              >
                {capturedPhotos.length > 0 ? "Done" : "Cancel"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
