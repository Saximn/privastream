"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [roomId, setRoomId] = useState("");
  const router = useRouter();

  // const createRoom = () => {
  //   router.push("/host");
  // };

  const enrollmentPage = () => {
    router.push("/enrollment");
  };

  const joinRoom = () => {
    if (roomId.trim()) {
      router.push(`/viewer/${roomId}`);
    }
  };

  return (
    <div>
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          {/* Hero Section */}
          <div className="mb-16">
            <div className="mx-auto mb-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-full w-fit">
              <div className="h-12 w-12 bg-black dark:bg-white rounded"></div>
            </div>
            <h1 className="text-4xl font-bold text-black dark:text-white mb-4">
              PrivaStream
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 mb-12 max-w-2xl mx-auto">
              Advanced virtual device app with face recognition, privacy
              protection, and streaming safety features.
            </p>
          </div>

          {/* Full Width Divider */}
          <div className="w-screen relative left-1/2 right-1/2 -ml-[50vw] -mr-[50vw] border-t border-gray-200 dark:border-gray-700 mb-12"></div>

          <div className="mb-16">
            {/* Main Action Cards - Equal Split Layout */}
            <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto mb-12">
              {/* Start Streaming Card */}
              <div className="group">
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700 h-full">
                  <div className="text-center h-full flex flex-col justify-between">
                    <div>
                      <div className="mx-auto mb-6 p-4 bg-gradient-to-br from-black to-gray-800 dark:from-white dark:to-gray-200 rounded-full w-fit shadow-lg">
                        <svg
                          className="h-12 w-12 text-white dark:text-black"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                          />
                        </svg>
                      </div>
                      <h2 className="text-2xl font-bold text-black dark:text-white mb-4">
                        Start Your Stream
                      </h2>
                      <p className="text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
                        Begin live streaming with advanced privacy protection,
                        real-time face recognition, and automatic content safety
                        features
                      </p>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-center space-x-2 text-sm text-gray-500 dark:text-gray-400 mb-4">
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-gray-600 dark:bg-gray-300 rounded-full"></div>
                          <span>Privacy Protected</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-gray-600 dark:bg-gray-300 rounded-full"></div>
                          <span>Face Recognition</span>
                        </div>
                      </div>
                      <Button
                        size="lg"
                        className="w-full bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
                        onClick={enrollmentPage}
                      >
                        Go Live Now
                        <svg
                          className="ml-2 h-5 w-5"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M14 5l7 7m0 0l-7 7m7-7H3"
                          />
                        </svg>
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Join Stream Card */}
              <div className="group">
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700 h-full">
                  <div className="text-center h-full flex flex-col justify-between">
                    <div>
                      <div className="mx-auto mb-6 p-4 bg-gradient-to-br from-gray-700 to-gray-900 dark:from-gray-200 dark:to-gray-400 rounded-full w-fit shadow-lg">
                        <svg
                          className="h-12 w-12 text-white dark:text-black"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                          />
                        </svg>
                      </div>
                      <h2 className="text-2xl font-bold text-black dark:text-white mb-4">
                        Join Stream
                      </h2>
                      <p className="text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
                        Connect to an existing stream using a room ID and enjoy
                        privacy-protected viewing with our advanced filtering
                        technology
                      </p>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-center space-x-2 text-sm text-gray-500 dark:text-gray-400 mb-4">
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-gray-600 dark:bg-gray-300 rounded-full"></div>
                          <span>Instant Access</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-gray-600 dark:bg-gray-300 rounded-full"></div>
                          <span>HD Quality</span>
                        </div>
                      </div>
                      <input
                        type="text"
                        placeholder="Enter Room ID"
                        value={roomId}
                        onChange={(e) => setRoomId(e.target.value)}
                        className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-gray-500 focus:border-transparent outline-none dark:border-gray-600 dark:bg-gray-800 dark:text-white text-center text-lg font-medium shadow-sm dark:focus:ring-gray-400 transition-all duration-300"
                      />
                      <Button
                        size="lg"
                        className="w-full bg-gray-800 text-white hover:bg-gray-900 dark:bg-gray-200 dark:text-black dark:hover:bg-gray-300 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={joinRoom}
                        disabled={!roomId.trim()}
                      >
                        Join Stream
                        <svg
                          className="ml-2 h-5 w-5"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
                          />
                        </svg>
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Full Width Divider */}
          <div className="w-screen relative left-1/2 right-1/2 -ml-[50vw] -mr-[50vw] border-t border-gray-200 dark:border-gray-700 mb-16"></div>

          {/* How it Works */}
          <div className="text-left">
            <h2 className="text-2xl font-bold text-center mb-8 text-black dark:text-white">
              How it Works
            </h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  1
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Enroll Your Face
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Capture your face securely to join the whitelist for clear
                  visibility during streams.
                </p>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  2
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Stream Safely
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Non-whitelisted faces are automatically blurred to protect
                  privacy and maintain safety.
                </p>
              </div>
              <div className="text-center">
                <div className="w-8 h-8 bg-black dark:bg-white text-white dark:text-black rounded-full flex items-center justify-center text-sm font-bold mx-auto mb-3">
                  3
                </div>
                <h3 className="font-semibold mb-2 text-black dark:text-white">
                  Monitor Safety
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Get detailed safety scores and analytics after each streaming
                  session.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
