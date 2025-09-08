import { NextRequest, NextResponse } from 'next/server';

const FACE_ENROLLMENT_API_URL = process.env.FACE_ENROLLMENT_API_URL || 'http://localhost:5003';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Validate required fields
    if (!body.frames || !body.room_id) {
      return NextResponse.json(
        { 
          success: false, 
          error: 'Missing required fields: frames and room_id',
          enrollment_complete: false
        },
        { status: 400 }
      );
    }

    // Forward the request to the Python Face Enrollment API
    const response = await fetch(`${FACE_ENROLLMENT_API_URL}/api/face-enrollment`, {
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
    console.error('Face enrollment API error:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Face enrollment service unavailable',
        enrollment_complete: false
      },
      { status: 503 }
    );
  }
}