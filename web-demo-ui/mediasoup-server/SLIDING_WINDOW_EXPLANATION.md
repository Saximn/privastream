# Sliding Window Audio Processing

## Overview
Updated the audio processing to use a sliding window approach that ensures words spanning chunk boundaries are properly detected.

## How It Works

### Before (Original Implementation):
```
Chunk 1: [  3 seconds  ] → Process → Output entire result
Chunk 2: [  3 seconds  ] → Process → Output entire result  
Chunk 3: [  3 seconds  ] → Process → Output entire result

Problem: Words spanning boundaries get cut off
```

### After (Sliding Window Implementation):
```
Chunk 1: [  3 seconds  ]           → Process [3s]        → Output entire result (3s)
Chunk 2: [  3 seconds  ]           → Process [3s + 3s]   → Output new portion only (3s)
Chunk 3: [  3 seconds  ]           → Process [3s + 3s]   → Output new portion only (3s)

Solution: Each processing includes previous chunk for context
```

## Processing Flow

### 1. First Chunk (roomId=room123)
- **Input**: Current chunk (3s)
- **Previous**: None
- **Processing**: Current chunk only
- **Output**: Entire processed result
- **Store**: Current chunk becomes "previous" for next iteration

### 2. Subsequent Chunks  
- **Input**: Current chunk (3s)
- **Previous**: Previous chunk (3s) 
- **Processing**: Combined [Previous + Current] (6s total)
- **Output**: Extract only second half (3s) to avoid duplicates
- **Store**: Current chunk becomes "previous" for next iteration

## Code Changes

### AudioRedactionProcessor Class Changes:

1. **Added sliding window storage**:
   ```javascript
   this.previousChunks = new Map(); // roomId -> previous 3-second chunk
   ```

2. **New processing method**:
   ```javascript
   async processAudioBufferSlidingWindow(roomId, currentChunk, previousChunk)
   ```

3. **Updated addAudioChunk method**:
   - Gets previous chunk for room
   - Processes with sliding window 
   - Stores current as previous for next iteration

4. **Enhanced cleanup**:
   - Clears previous chunk storage when room is cleaned up

## Benefits

✅ **Word Boundary Detection**: Words spanning chunks are now detected properly  
✅ **No Audio Duplication**: Only new portions are output to avoid duplicates  
✅ **Backward Compatible**: All existing delay/timing logic preserved  
✅ **Atomic Changes**: Minimal changes to existing codebase  
✅ **Per-Room Isolation**: Each room maintains its own sliding window state  

## Timing Synchronization Fix

### The Problem
With sliding window, the output audio represents content from 3 seconds earlier:
- We process [Previous 3s + Current 3s] 
- But only output the second half (representing previous chunk's timeframe)
- This created a 3-second delay compared to video

### The Solution
```javascript
// Calculate timing offset for sliding window
const slidingWindowOffset = result.metadata?.hadPreviousChunk ? 3000 : 0;
const effectiveChunkStartTime = chunkStartTime - slidingWindowOffset;

// Maintain 8-second total delay with corrected timing
const targetOutputTime = effectiveChunkStartTime + 8000;
const delayNeeded = Math.max(0, targetOutputTime - Date.now());
```

### How It Works
- **First chunk**: No offset (hadPreviousChunk = false) → Normal timing
- **Subsequent chunks**: 3-second offset (hadPreviousChunk = true) → Compensated timing
- **Result**: Audio and video remain synchronized at 8-second total delay

## Preserved Functionality

- ✅ 3-second client-side chunking unchanged
- ✅ 8-second total delay maintained with sync compensation
- ✅ Audio/video synchronization preserved
- ✅ Error handling unchanged
- ✅ All existing logging and metadata enhanced

## Example Scenario

**Word "CONFIDENTIAL" spans chunks**:
- Chunk 1: "This document is CONFIDEN..." 
- Chunk 2: "...TIAL and should not..."

**Before**: "CONFIDEN" and "TIAL" processed separately → May miss the full word  
**After**: Chunk 2 processes "CONFIDEN" + "TIAL" together → Detects full "CONFIDENTIAL"

The sliding window ensures complete context for accurate PII detection while maintaining all existing performance and timing characteristics.