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
  const whitelistedCount = faces.filter((face) => face.whitelisted).length;
  const blurredCount = faces.length - whitelistedCount;

  return (
    <Card className="w-full bg-white dark:bg-black dark:border-white shadow-lg">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
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

          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className="bg-white dark:bg-black dark:border-white"
            >
              <Eye className="w-3 h-3 mr-1 text-black dark:text-white" />
              {whitelistedCount} visible
            </Badge>
            <Badge
              variant="outline"
              className="bg-white dark:bg-black dark:border-white"
            >
              <Eye className="w-3 h-3 mr-1 text-black dark:text-white" />
              {blurredCount} blurred
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        {faces.length > 0 ? (
          <div
            className={`space-y-2 overflow-y-auto pr-2 ${
              faces.length > 2 ? "max-h-[200px]" : ""
            }`}
          >
            {faces.map((face, index) => (
              <div
                key={face.id}
                className={`group relative p-3 border-2 rounded-lg transition-all duration-200 cursor-pointer hover:shadow-md ${
                  face.whitelisted
                    ? "bg-white dark:bg-black border-black dark:border-white hover:bg-gray-50 dark:hover:bg-gray-900"
                    : "bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
                onClick={() => onToggleWhitelist?.(face.id)}
              >
                <div className="flex items-center gap-3">
                  {/* Index Badge */}
                  <div className="flex-shrink-0">
                    <div className="relative">
                      <div className="w-6 h-6 bg-black dark:bg-white text-white dark:text-black flex items-center justify-center text-xs font-bold rounded-md shadow-sm">
                        {index + 1}
                      </div>
                    </div>
                  </div>

                  {/* Avatar with enhanced styling */}
                  <div className="flex-shrink-0 relative">
                    <Avatar className="w-10 h-10 border-2 dark:border-white shadow-sm">
                      <AvatarImage
                        src={face.image}
                        alt={face.name}
                        className={`object-cover ${
                          face.whitelisted ? "" : "blur-sm"
                        }`}
                      />
                      <AvatarFallback className="bg-white dark:bg-black text-black dark:text-white font-bold text-sm border border-black dark:border-white">
                        {face.name.charAt(0).toUpperCase()}
                      </AvatarFallback>
                    </Avatar>

                    {/* Privacy overlay icon */}
                    <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-white dark:bg-black rounded-full flex items-center justify-center shadow-sm border border-black dark:border-white">
                      {face.whitelisted ? (
                        <Shield className="w-2 h-2 text-black dark:text-white" />
                      ) : (
                        <ShieldOff className="w-2 h-2 text-gray-600 dark:text-gray-400" />
                      )}
                    </div>
                  </div>

                  {/* Face Info */}
                  <div className="flex-grow min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-black dark:text-white text-sm truncate">
                        {face.name}
                      </h4>
                      <Badge
                        variant={face.whitelisted ? "default" : "outline"}
                        className={`text-xs font-medium py-0 px-1 h-5 ${
                          face.whitelisted
                            ? "bg-black dark:bg-white text-white dark:text-black border-black dark:border-white"
                            : "bg-white dark:bg-black text-black dark:text-white border-black dark:border-white"
                        }`}
                      >
                        {face.whitelisted ? "VISIBLE" : "PROTECTED"}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Privacy:{" "}
                      <span className="font-medium text-black dark:text-white">
                        {face.whitelisted ? "Off" : "On"}
                      </span>
                    </p>
                  </div>

                  {/* Action indicator */}
                  <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <div className="w-6 h-6 bg-gray-200 dark:bg-gray-700 border border-gray-400 dark:border-gray-600 rounded flex items-center justify-center">
                      {face.whitelisted ? (
                        <EyeOff className="w-3 h-3 text-black dark:text-white" />
                      ) : (
                        <Eye className="w-3 h-3 text-black dark:text-white" />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Enhanced Empty State */
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-3 bg-white dark:bg-black border-2 border-black dark:border-white flex items-center justify-center rounded-xl shadow-inner">
              <Users className="w-8 h-8 text-black dark:text-white" />
            </div>
            <h4 className="text-base font-semibold text-black dark:text-white mb-2">
              No Faces Detected
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 max-w-xs mx-auto">
              Point your camera towards faces to enable real-time privacy
              protection
            </p>
            <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-500 dark:text-gray-500">
              <div className="w-2 h-2 bg-black dark:bg-white rounded-full animate-pulse" />
              <span>Scanning for faces...</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
