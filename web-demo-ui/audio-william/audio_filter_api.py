import torch
import soundfile as sf
import numpy as np
import librosa
from transformers import pipeline

# -----------------------------
# 1️⃣ Load Models
# -----------------------------
# NER pipeline (DeBERTa or BERT NER)
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Keyword fallback
SENSITIVE_KEYWORDS = ["password", "secret", "pin"]

# ASR pipeline (Whisper)
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# -----------------------------
# 2️⃣ ASR Function
# -----------------------------
def transcribe_audio(audio_file):
    """
    Returns transcript, word-level timestamps, audio array, sample rate
    """
    y, sr = librosa.load(audio_file, sr=16000)  # Whisper expects 16kHz
    result = asr_pipeline(y, chunk_length_s=30, return_timestamps="word")

    transcript = result["text"]
    words = result["chunks"]  # 'text' and 'timestamp' for each chunk
    word_times = []
    current_pos = 0
    for w in words:
        text = w["text"].strip()
        start_time = w["timestamp"][0]
        end_time = w["timestamp"][1]
        for token in text.split():
            char_start = transcript.find(token, current_pos)
            char_end = char_start + len(token)
            current_pos = char_end
            word_times.append({
                "word": token,
                "start": start_time,
                "end": end_time,
                "char_start": char_start,
                "char_end": char_end
            })
    return transcript, word_times, y, sr

# -----------------------------
# 3️⃣ Detect Sensitive Words / NER
# -----------------------------
def detect_sensitive_spans(transcript, word_times):
    redact_intervals = []

    # --- Keyword matching ---
    for idx, w in enumerate(word_times):
        if w["word"].lower() in SENSITIVE_KEYWORDS:
            start = w["start"]
            end = word_times[min(idx+5, len(word_times)-1)]["end"]  # 5-word lookahead
            redact_intervals.append((start, end))

    # --- NER-based ---
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

    # --- Merge overlapping intervals ---
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
        audio_copy[i1:i2] = 0.0  # mute
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
