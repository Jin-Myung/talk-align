import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import signal

# ================================
# 1. Load script file
# ================================
if len(sys.argv) < 2:
    print("Usage: python vad_align.py <script.txt>")
    sys.exit(1)

script_file = sys.argv[1]
with open(script_file, "r", encoding="utf-8") as f:
    script_lines = [line.strip() for line in f if line.strip()]

print(f"Script loaded ({len(script_lines)} lines)")

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
                # Speech ended → run STT
                audio = np.frombuffer(b"".join(buffer), np.int16).astype(np.float32) / 32768.0
                segments, _ = model.transcribe(audio, language="ko", beam_size=1)
                text = "".join(seg.text for seg in segments).strip()

                if text:
                    # Compare with script (search current line ~ next 3 lines)
                    start = current_index
                    end = min(len(script_lines), current_index + 4)
                    search_range = script_lines[start:end]

                    match, score, idx_rel = process.extractOne(
                        text, search_range, scorer=fuzz.partial_ratio
                    )
                    idx = start + idx_rel

                    if score >= threshold:
                        current_index = idx
                        print(f"\nRecognized: {text}")
                        print(f"Script[{idx}]: {match} (similarity {score}%)")
                    else:
                        print(f"\nRecognized: {text}")
                        print(f"No matching script (similarity {score}%)")

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
t1.join()
t2.join()
print("Shutdown complete")
