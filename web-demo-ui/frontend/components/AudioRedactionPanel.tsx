'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Shield, 
  ShieldAlert, 
  Mic, 
  MicOff, 
  Eye, 
  EyeOff,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  Clock,
  Users,
  Volume2,
  VolumeX
} from 'lucide-react';
import { useAudioRedaction, RedactionResult, PIIDetection } from '@/lib/audio-redaction-client';

interface AudioRedactionPanelProps {
  roomId: string;
  role: 'host' | 'viewer';
  mediaStream?: MediaStream;
  className?: string;
}

const PIITypeColors: { [key: string]: string } = {
  'PERSON': 'bg-red-100 text-red-800',
  'EMAIL': 'bg-orange-100 text-orange-800',
  'PHONE_NUMBER': 'bg-yellow-100 text-yellow-800',
  'ADDRESS': 'bg-blue-100 text-blue-800',
  'CREDIT_CARD': 'bg-purple-100 text-purple-800',
  'SSN': 'bg-pink-100 text-pink-800',
  'OTHER': 'bg-gray-100 text-gray-800'
};

const PIITypeIcons: { [key: string]: React.ReactNode } = {
  'PERSON': <Users className="h-3 w-3" />,
  'EMAIL': <Shield className="h-3 w-3" />,
  'PHONE_NUMBER': <Shield className="h-3 w-3" />,
  'ADDRESS': <Shield className="h-3 w-3" />,
  'CREDIT_CARD': <ShieldAlert className="h-3 w-3" />,
  'SSN': <ShieldAlert className="h-3 w-3" />,
  'OTHER': <AlertTriangle className="h-3 w-3" />
};

export default function AudioRedactionPanel({ 
  roomId, 
  role, 
  mediaStream, 
  className = '' 
}: AudioRedactionPanelProps) {
  const {
    client,
    isConnected,
    redactionResults,
    piiDetections,
    stats,
    initializeClient,
    connect,
    disconnect,
    joinRoom,
    startAudioCapture,
    stopAudioCapture,
    updateStats
  } = useAudioRedaction();

  const [isEnabled, setIsEnabled] = useState(true);
  const [isCapturing, setIsCapturing] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [recentAlerts, setRecentAlerts] = useState<RedactionResult[]>([]);

  // Initialize client on mount
  useEffect(() => {
    const serviceUrl = process.env.NEXT_PUBLIC_REDACTION_SERVICE_URL || 'http://localhost:5002';
    initializeClient(serviceUrl);
  }, [initializeClient]);

  // Connect and join room
  useEffect(() => {
    if (client && roomId) {
      const setupConnection = async () => {
        try {
          setConnectionError(null);
          await connect();
          await joinRoom(roomId, role);
        } catch (error) {
          console.error('Failed to setup audio redaction:', error);
          setConnectionError(error instanceof Error ? error.message : 'Connection failed');
        }
      };

      setupConnection();

      return () => {
        disconnect();
      };
    }
  }, [client, roomId, role, connect, disconnect, joinRoom]);

  // Handle audio capture
  useEffect(() => {
    if (mediaStream && isConnected && isEnabled && role === 'host') {
      if (isCapturing) {
        startAudioCapture(mediaStream);
      } else {
        stopAudioCapture();
      }
    }

    return () => {
      stopAudioCapture();
    };
  }, [mediaStream, isConnected, isEnabled, isCapturing, role, startAudioCapture, stopAudioCapture]);

  // Update recent alerts
  useEffect(() => {
    if (piiDetections.length > 0) {
      setRecentAlerts(prev => [...piiDetections.slice(-5)]);
    }
  }, [piiDetections]);

  const toggleCapture = () => {
    if (role === 'host') {
      setIsCapturing(!isCapturing);
    }
  };

  const toggleEnabled = () => {
    setIsEnabled(!isEnabled);
    if (!isEnabled && role === 'host') {
      setIsCapturing(true);
    } else {
      setIsCapturing(false);
    }
  };

  const getStatusColor = () => {
    if (!isConnected) return 'bg-red-500';
    if (!isEnabled) return 'bg-gray-500';
    if (isCapturing) return 'bg-green-500';
    return 'bg-yellow-500';
  };

  const getStatusText = () => {
    if (!isConnected) return 'Disconnected';
    if (!isEnabled) return 'Disabled';
    if (isCapturing) return 'Active';
    return 'Ready';
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main Status Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5 text-blue-600" />
            Audio PII Protection
            <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
          </CardTitle>
          <Badge variant={isConnected ? 'default' : 'destructive'}>
            {getStatusText()}
          </Badge>
        </CardHeader>
        <CardContent>
          {connectionError && (
            <Alert className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Connection Error: {connectionError}
              </AlertDescription>
            </Alert>
          )}

          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Button
                variant={isEnabled ? 'default' : 'outline'}
                size="sm"
                onClick={toggleEnabled}
                disabled={!isConnected}
              >
                {isEnabled ? (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    Protection On
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 mr-2" />
                    Protection Off
                  </>
                )}
              </Button>

              {role === 'host' && (
                <Button
                  variant={isCapturing ? 'destructive' : 'outline'}
                  size="sm"
                  onClick={toggleCapture}
                  disabled={!isConnected || !isEnabled}
                >
                  {isCapturing ? (
                    <>
                      <MicOff className="h-4 w-4 mr-2" />
                      Stop Capture
                    </>
                  ) : (
                    <>
                      <Mic className="h-4 w-4 mr-2" />
                      Start Capture
                    </>
                  )}
                </Button>
              )}
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDetails(!showDetails)}
            >
              {showDetails ? (
                <>
                  <EyeOff className="h-4 w-4 mr-2" />
                  Hide Details
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Show Details
                </>
              )}
            </Button>
          </div>

          {/* Quick Stats */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {stats.processed_segments}
                </div>
                <div className="text-xs text-gray-500">Segments Processed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {stats.total_pii_detections}
                </div>
                <div className="text-xs text-gray-500">PII Detected</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {stats.average_processing_time.toFixed(2)}s
                </div>
                <div className="text-xs text-gray-500">Avg Process Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {recentAlerts.length}
                </div>
                <div className="text-xs text-gray-500">Recent Alerts</div>
              </div>
            </div>
          )}

          {/* Recent PII Alerts */}
          {recentAlerts.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-semibold text-sm flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                Recent PII Detections
              </h4>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {recentAlerts.map((alert, index) => (
                  <div
                    key={`${alert.segment_id}-${index}`}
                    className="p-2 bg-yellow-50 border border-yellow-200 rounded-lg"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Clock className="h-3 w-3 text-gray-500" />
                        <span className="text-xs text-gray-600">
                          {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                      <Badge variant="destructive" className="text-xs">
                        {alert.pii_count} PII detected
                      </Badge>
                    </div>
                    {role === 'host' && alert.pii_detections && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {alert.pii_detections.map((detection, detIndex) => (
                          <Badge
                            key={detIndex}
                            className={`text-xs ${PIITypeColors[detection.pii_type] || PIITypeColors.OTHER}`}
                          >
                            {PIITypeIcons[detection.pii_type]}
                            {detection.pii_type} ({(detection.confidence * 100).toFixed(0)}%)
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed View */}
      {showDetails && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Detailed Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Processing Activity */}
              {redactionResults.length > 0 && (
                <div>
                  <h4 className="font-semibold text-sm mb-2 flex items-center gap-2">
                    <Volume2 className="h-4 w-4 text-blue-600" />
                    Recent Processing Activity
                  </h4>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {redactionResults.slice(-10).map((result, index) => (
                      <div
                        key={`${result.segment_id}-${index}`}
                        className="p-3 bg-blue-50 border border-blue-200 rounded-lg"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">
                            Segment {result.segment_id}
                          </span>
                          <div className="flex items-center gap-2">
                            {result.pii_count > 0 ? (
                              <VolumeX className="h-4 w-4 text-red-500" />
                            ) : (
                              <Volume2 className="h-4 w-4 text-green-500" />
                            )}
                            <span className="text-xs text-gray-500">
                              {result.processing_time.toFixed(2)}s
                            </span>
                          </div>
                        </div>
                        
                        {role === 'host' && (
                          <div className="space-y-1">
                            <div className="text-xs text-gray-600">
                              <strong>Original:</strong> {result.original_transcription}
                            </div>
                            {result.pii_count > 0 && (
                              <div className="text-xs text-gray-600">
                                <strong>Redacted:</strong> {result.redacted_transcription}
                              </div>
                            )}
                          </div>
                        )}
                        
                        {result.pii_count > 0 && (
                          <div className="mt-2">
                            <Badge variant="destructive" className="text-xs">
                              {result.pii_count} PII redacted
                            </Badge>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Connection Info */}
              <div>
                <h4 className="font-semibold text-sm mb-2">Connection Info</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Status:</span>
                    <span className="ml-2 font-medium">
                      {isConnected ? (
                        <span className="text-green-600 flex items-center gap-1">
                          <CheckCircle className="h-3 w-3" />
                          Connected
                        </span>
                      ) : (
                        <span className="text-red-600 flex items-center gap-1">
                          <XCircle className="h-3 w-3" />
                          Disconnected
                        </span>
                      )}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Role:</span>
                    <span className="ml-2 font-medium capitalize">{role}</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}