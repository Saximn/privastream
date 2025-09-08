import { NextRequest, NextResponse } from 'next/server';

const FACE_ENROLLMENT_API_URL = process.env.FACE_ENROLLMENT_API_URL || 'http://localhost:5003';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Forward the request to the Python Face Enrollment API
    const response = await fetch(`${FACE_ENROLLMENT_API_URL}/api/face-detection`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Face detection API error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Face detection service unavailable',
        faces_detected: []
      },
      { status: 503 }
    );
  }
}