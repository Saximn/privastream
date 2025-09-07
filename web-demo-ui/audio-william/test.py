import torch
import soundfile as sf
import numpy as np
from transformers import pipeline
import librosa

# -----------------------------
# 1️⃣ Load Pretrained Models
# -----------------------------
# NER pipeline for detecting sensitive words (out-of-the-box)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Optional keyword list for additional detection
SENSITIVE_KEYWORDS = ["password", "secret", "pin", "address"]

# -----------------------------
# 2️⃣ Audio → ASR (Whisper)
# -----------------------------
# Use HuggingFace Whisper pipeline for speech-to-text
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(audio_file):
    """
    Returns: transcript string, word-level timestamps (list of dicts)
    """
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)  # Whisper expects 16kHz
    # Run ASR
    result = asr_pipeline(y, chunk_length_s=30, return_timestamps="word")
    transcript = result["text"]
    words = result["chunks"]  # each chunk has 'text', 'timestamp' keys
    word_times = []
    for w in words:
        # Some chunks may be multiple words
        text = w["text"].strip()
        start = w["timestamp"][0]
        end = w["timestamp"][1]
        for token in text.split():
            word_times.append({"word": token, "start": start, "end": end})
    return transcript, word_times, y, sr

# -----------------------------
# 3️⃣ Detect Sensitive Words
# -----------------------------
def detect_sensitive_spans(transcript, word_times):
    """
    Returns list of (start_sec, end_sec) intervals to mute
    """
    redact_intervals = []

    # Build ASR word char positions
    current_pos = 0
    for w in word_times:
        w["char_start"] = current_pos
        w["char_end"] = current_pos + len(w["word"])
        current_pos = w["char_end"] + 1  # add 1 for space

    # 1️⃣ Keyword matching (keep your existing)
    for idx, w in enumerate(word_times):
        if w["word"].lower() in SENSITIVE_KEYWORDS:
            start = w["start"]
            end = word_times[min(idx+5, len(word_times)-1)]["end"]
            redact_intervals.append((start, end))

    # 2️⃣ NER spans
    ner_results = ner_pipeline(transcript)
    for entity in ner_results:
        if entity["entity_group"] in ["PER", "LOC", "MISC"]:  # treat as sensitive
            start_char = entity["start"]
            end_char = entity["end"]

            overlapping_words = [
                w for w in word_times
                if not (w["char_end"] <= start_char or w["char_start"] >= end_char)
            ]

            if overlapping_words:
                start_sec = min(w["start"] for w in overlapping_words)
                end_sec = max(w["end"] for w in overlapping_words)
                redact_intervals.append((start_sec, end_sec))

    # Merge overlapping intervals
    if not redact_intervals:
        return []

    redact_intervals = sorted(redact_intervals, key=lambda x: x[0])
    merged = [redact_intervals[0]]
    for start, end in redact_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged

# -----------------------------
# 4️⃣ Redact Audio
# -----------------------------
def mute_intervals(audio, sr, intervals):
    audio_copy = audio.copy()
    for start, end in intervals:
        i1 = int(start * sr)
        i2 = int(end * sr)
        audio_copy[i1:i2] = 0.0
    return audio_copy

# -----------------------------
# 5️⃣ Main Function
# -----------------------------
def redact_audio(input_wav, output_wav):
    print("[*] Transcribing audio...")
    transcript, word_times, audio, sr = transcribe_audio(input_wav)
    print("Transcript:", transcript)

    print("[*] Detecting sensitive words...")
    intervals = detect_sensitive_spans(transcript, word_times)
    print("Redact intervals (sec):", intervals)

    print("[*] Redacting audio...")
    redacted_audio = mute_intervals(audio, sr, intervals)

    print("[*] Saving redacted audio to", output_wav)
    sf.write(output_wav, redacted_audio, sr)
    print("[*] Done!")

# -----------------------------
# 6️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    input_file = "input.wav"
    output_file = "output_redacted.wav"
    redact_audio(input_file, output_file)
