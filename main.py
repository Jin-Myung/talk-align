import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import signal, re

# ================================
# 1. Load script file
# ================================
if len(sys.argv) < 2:
    print("Usage: python vad_align.py <script.txt>")
    sys.exit(1)

script_file = sys.argv[1]
with open(script_file, "r", encoding="utf-8") as f:
    script_lines = [line.strip() for line in f if line.strip()]

# ================================
# 2. Initialize model / audio / VAD
# ================================
model = WhisperModel("small", device="cpu", compute_type="int8")

RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 480 samples
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)   # VAD sensitivity (0=permissive, 3=aggressive)

audio_q = queue.Queue()
running = True
current_index = 0  # current script line index

# ================================
# Min chunk length thresholds
# ================================
MIN_CHARS_FOR_MATCH = 6      # require at least 6 significant chars
MIN_MS_FOR_MATCH = 1000      # and at least 1 sec of speech

# keep letters/digits/Hangul when counting significant chars
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

# ================================
# 3. Audio capture thread
# ================================
def audio_capture():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while running:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            audio_q.put(frame)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# ================================
# 4. VAD + STT + Script alignment
# ================================
def vad_loop():
    global current_index
    buffer = []
    is_speaking = False
    threshold = 70  # similarity threshold

    # NEW: pending buffers for AND condition
    pending_text = ""
    pending_ms = 0

    while running or not audio_q.empty():
        try:
            frame = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if len(frame) != FRAME_SIZE * 2:
            continue

        if vad.is_speech(frame, RATE):
            buffer.append(frame)
            is_speaking = True
        else:
            if is_speaking and buffer:
                # Speech ended → run STT for this utterance
                audio = np.frombuffer(b"".join(buffer), np.int16).astype(np.float32) / 32768.0
                utter_ms = int(len(audio) / RATE * 1000)

                segments, _ = model.transcribe(audio, language="ko", beam_size=1)
                raw_text = "".join(seg.text for seg in segments).strip()

                if raw_text:
                    # AND condition buffering: combine until both thresholds met
                    combined_text = (pending_text + " " + raw_text).strip() if pending_text else raw_text
                    combined_ms = pending_ms + utter_ms

                    if significant_len(combined_text) >= MIN_CHARS_FOR_MATCH and combined_ms >= MIN_MS_FOR_MATCH:
                        # thresholds satisfied → align now
                        print(f"\nRecognized: {combined_text}")

                        start = current_index
                        end = min(len(script_lines), current_index + 4)  # current .. current+3
                        search_range = script_lines[start:end]

                        if search_range:
                            match, score, idx_rel = process.extractOne(
                                combined_text, search_range, scorer=fuzz.partial_ratio
                            )
                            idx = start + idx_rel

                            if score >= threshold:
                                current_index = idx
                                print(f"[{idx}]: {match} (similarity {score}%)")
                            else:
                                print(f"No matching script (similarity {score}%)")
                        else:
                            print("No matching script (end of script window)")

                        # clear pending after alignment
                        pending_text = ""
                        pending_ms = 0
                    else:
                        # keep buffering
                        pending_text = combined_text
                        pending_ms = combined_ms
                        print(f"\nRecognized (buffering): {raw_text} | "
                              f"chars={significant_len(pending_text)}/{MIN_CHARS_FOR_MATCH}, "
                              f"ms={pending_ms}/{MIN_MS_FOR_MATCH}")

                buffer = []
                is_speaking = False

    print("VAD loop stopped")

# ================================
# 5. Graceful shutdown
# ================================
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# ================================
# 6. Run
# ================================
t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, daemon=True)

t1.start()
t2.start()
print(f"Ready to align your speech with the script ({len(script_lines)} lines). Press Ctrl+C to stop.")

t1.join()
t2.join()
print("Shutdown complete")
