"use client";
import { useState } from "react";
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
import { CameraCapture } from "@/components/enrollment/cameracapture";
import { FacePreview } from "@/components/enrollment/facepreview";
import { useRouter } from "next/navigation";

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

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      fullName: "",
      email: "",
    },
  });

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
    // Get the enrolled room ID from the CameraCapture component
    const enrolledRoomId = localStorage.getItem('enrolledRoomId');
    if (enrolledRoomId) {
      // Pass the room ID to the host page
      sessionStorage.setItem('faceEnrolledRoomId', enrolledRoomId);
      router.push("/host");
    } else {
      router.push("/host");
    }
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
                <CameraCapture onPhotosChange={setCapturedPhotos} />
              </CardContent>
              <div className="flex justify-between px-6 pb-6 mt-auto">
                <Button
                  variant="outline"
                  onClick={() => setEnrollmentStep("intro")}
                  className="border-gray-300 text-black hover:bg-gray-100 dark:border-gray-600 dark:text-white dark:hover:bg-gray-800"
                >
                  Back
                </Button>
                <Button
                  onClick={() => setEnrollmentStep("form")}
                  disabled={capturedPhotos.length === 0}
                  className="bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
                >
                  Continue →
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
