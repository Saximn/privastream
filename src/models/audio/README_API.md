# Audio Redaction Flask API

A Flask API server that processes audio chunks for PII (Personally Identifiable Information) redaction using speech-to-text and NER (Named Entity Recognition) models.

## Features

- **Real-time audio processing**: Processes audio chunks and returns redacted audio
- **PII Detection**: Uses BERT-based NER to detect sensitive information
- **Speech Recognition**: Uses OpenAI Whisper for audio transcription  
- **Audio Redaction**: Mutes detected sensitive audio segments
- **REST API**: JSON and binary endpoints for integration
- **Cross-Origin Support**: CORS enabled for web integration

## Setup

1. **Install Dependencies**:
   ```bash
   cd audio-william
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python audio_redaction_server.py
   ```
   
   The server will start on `http://localhost:5002`

## API Endpoints

### Health Check
```
GET /health
```
Returns server status and model loading status.

### Process Audio (JSON)
```
POST /process_audio
Content-Type: application/json

{
  "audio_data": "<base64-encoded-audio>",
  "sample_rate": 16000
}
```

**Response**:
```json
{
  "success": true,
  "redacted_audio_data": "<base64-encoded-redacted-audio>",
  "transcript": "Hello world",
  "pii_count": 0,
  "redacted_intervals": [],
  "sample_rate": 16000,
  "timestamp": "2025-09-06T13:50:14.123Z"
}
```

### Process Audio (Raw Binary)
```
POST /process_audio_raw
Content-Type: application/octet-stream
X-Sample-Rate: 16000

<raw-audio-bytes>
```

Returns raw redacted audio bytes with processing info in headers.

## Audio Format

- **Sample Rate**: 16000 Hz (recommended)
- **Format**: 16-bit PCM mono
- **Encoding**: Little-endian for binary, base64 for JSON

## Integration with Mediasoup

The API is designed to work with the existing Mediasoup audio redaction plugin. Here's the flow:

1. **Mediasoup** captures audio from streamer
2. **Audio chunks** are sent to Flask API on port 5002
3. **API processes** audio for PII detection and redaction
4. **Redacted audio** is returned to Mediasoup
5. **Clean audio** is streamed to viewers

## Testing

Run the test suite:
```bash
python simple_test.py
```

## Configuration

The server uses these models by default:
- **NER**: `dslim/bert-base-NER` 
- **ASR**: `openai/whisper-small`

Sensitive keywords: `["password", "secret", "pin", "ssn", "social", "credit", "card"]`

## Performance

- **Processing Time**: ~6 seconds for 2-second audio clips
- **Latency**: Acceptable for streaming with buffering
- **Memory**: Models loaded once at startup
- **CPU**: Runs on CPU, GPU support available

## Production Notes

- Use a production WSGI server (gunicorn, uWSGI)
- Consider model optimization for faster processing
- Implement proper logging and monitoring
- Add authentication if needed
- Scale with multiple workers for higher throughput