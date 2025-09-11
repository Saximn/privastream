"use client";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye, EyeOff, Users, Shield, ShieldOff } from "lucide-react";

type Face = {
  id: string;
  name: string;
  image: string;
  whitelisted: boolean;
};

interface FacePreviewProps {
  faces: Face[];
  onToggleWhitelist?: (faceId: string) => void;
}

export function FacePreview({ faces, onToggleWhitelist }: FacePreviewProps) {
  const whitelistedFace = faces.find((face) => face.whitelisted);

  return (
    <Card className="w-full bg-white dark:bg-black dark:border-white shadow-lg">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-black dark:bg-white rounded-lg">
            <Users className="w-5 h-5 text-white dark:text-black" />
          </div>
          <div>
            <h3 className="font-bold text-lg text-black dark:text-white">
              Face Recognition
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Live detection & privacy controls
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {whitelistedFace ? (
          <div className="flex flex-row items-center justify-center py-6 gap-6">
            <div className="flex-shrink-0 flex items-center justify-center">
              <Avatar className="w-32 h-32 border-2 dark:border-white shadow-lg">
                <AvatarImage
                  src={whitelistedFace.image}
                  alt={whitelistedFace.name}
                  className="object-cover"
                />
                <AvatarFallback className="bg-white dark:bg-black text-black dark:text-white font-bold text-3xl border border-black dark:border-white">
                  {whitelistedFace.name.charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
            </div>
            <div className="flex flex-col items-start justify-center">
              <h4 className="font-semibold text-xl text-black dark:text-white mb-3">
                {whitelistedFace.name}
              </h4>
              <Badge
                variant="default"
                className="text-lg font-bold py-1 px-3 h-8 bg-black dark:bg-white text-white dark:text-black border-black dark:border-white mb-3"
              >
                VISIBLE
              </Badge>
              <div className="mt-2">
                <span className="text-lg font-bold text-red-600 dark:text-red-400">Only this face will be visible</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-3 bg-white dark:bg-black border-2 border-black dark:border-white flex items-center justify-center rounded-xl shadow-inner">
              <Users className="w-8 h-8 text-black dark:text-white" />
            </div>
            <h4 className="text-base font-semibold text-black dark:text-white mb-2">
              No Faces Whitelisted
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 max-w-xs mx-auto">
              Select a face to make it visible
            </p>
            <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-500 dark:text-gray-500">
              <div className="w-2 h-2 bg-black dark:bg-white rounded-full animate-pulse" />
              <span>Waiting for whitelist...</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
