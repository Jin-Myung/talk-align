import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import signal, re

# ================================
# Load script file
# ================================
if len(sys.argv) < 2:
    print("Usage: python vad_align.py <script.txt>")
    sys.exit(1)

script_file = sys.argv[1]
with open(script_file, "r", encoding="utf-8") as f:
    script_lines = [line.strip() for line in f if line.strip()]

# ================================
# Initialize model / audio / VAD
# ================================
model = WhisperModel("small", device="cpu", compute_type="int8")

RATE = 16000
FRAME_DURATION = 30  # 30 ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 480 samples
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)   # VAD sensitivity (0=permissive, 3=aggressive)

rt_audio_q = queue.Queue()
running = True

# ================================
# Utterance length thresholds for script matching
# ================================
MIN_UTTER_TEXT_LEN_FOR_MATCH = 6      # require at least 6 significant chars
MIN_UTTER_LEN_IN_MS_FOR_MATCH = 1000  # and at least 1 sec of speech
MAX_UTTER_LEN_IN_MS = 10000  # truncate utterance to 10 sec in case of no silence

# ================================
# Script matching threshold
# ================================
SCRIPT_MATCH_THRESHOLD = 70  # require at least 70% similarity

# ================================
# Audio capture thread
# ================================
def audio_capture():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while running:
            frame = stream.read(FRAME_SIZE)
            rt_audio_q.put(frame)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# ================================
# Keep letters/digits/Hangul when counting significant chars
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

# ================================
# Process utterance: STT + script alignment
# ================================
def process_utterance(utter, pending_utter_text, pending_utter_len, current_line_in_script):
    normalized_utter = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(normalized_utter) / RATE * 1000)

    segments, _ = model.transcribe(normalized_utter, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if raw_text:
        combined_text = (pending_utter_text + " " + raw_text).strip() if pending_utter_text else raw_text
        combined_ms = pending_utter_len + utter_ms

        if significant_len(combined_text) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and combined_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:

            print(f"\nRecognized: {combined_text}")

            start = max(0, current_line_in_script - 1)
            end = min(len(script_lines), current_line_in_script + 4)
            search_range = script_lines[start:end]

            if search_range:
                match, score, idx_rel = process.extractOne(
                    combined_text, search_range, scorer=fuzz.partial_ratio
                )
                idx = start + idx_rel

                if score >= SCRIPT_MATCH_THRESHOLD:
                    current_line_in_script = idx
                    print(f"[{idx}]: {match} (similarity {score:.3f}%)")
                else:
                    print(f"No matching script (similarity {score:.3f}%)")
            else:
                print("No matching script (end of script window)")

            pending_utter_text = ""
            pending_utter_len = 0
        else:
            pending_utter_text = combined_text
            pending_utter_len = combined_ms
            print(f"\nToo short utterance: {raw_text} | "
                  f"chars={significant_len(pending_utter_text)}/{MIN_UTTER_TEXT_LEN_FOR_MATCH}, "
                  f"ms={pending_utter_len}/{MIN_UTTER_LEN_IN_MS_FOR_MATCH}")

    return pending_utter_text, pending_utter_len, current_line_in_script

# ================================
# VAD + utterance processing loop
# ================================
def vad_loop():
    current_line_in_script = 0
    utter = []
    in_utter = False

    # If an utterance is too short, we buffer it and prepend to the next one rather than matching a script line.
    pending_utter_text = ""
    pending_utter_len = 0  # in ms

    # VAD hangover settings
    speech_start_frames = 2  # 2 * FRAME_DURATION = 90ms to confirm speech start
    silence_end_frames = 6   # 6 * FRAME_DURATION = 180ms to confirm end
    speech_count = 0
    silence_count = 0

    while running:
        try:
            frame = rt_audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if len(frame) < FRAME_SIZE * 2:
            continue

        is_speech = vad.is_speech(frame, RATE)

        if not in_utter:
            if is_speech:
                speech_count += 1
                if speech_count >= speech_start_frames:
                    in_utter = True
                    utter = []
                    utter.append(frame)
                    silence_count = 0
            else:
                speech_count = 0

        else:
            utter.append(frame)
            if is_speech:
                silence_count = 0
                if len(utter) * FRAME_DURATION >= MAX_UTTER_LEN_IN_MS:
                    # Force process if utterance too long
                    pending_utter_text, pending_utter_len, current_line_in_script = process_utterance(
                        utter, pending_utter_text, pending_utter_len, current_line_in_script
                    )
                    # reset for next utterance
                    in_utter = False
                    speech_count = 0
                    utter = []

            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_utter_text, pending_utter_len, current_line_in_script = process_utterance(
                        utter, pending_utter_text, pending_utter_len, current_line_in_script
                    )
                    # reset for next utterance
                    in_utter = False
                    speech_count = 0
                    silence_count = 0
                    utter = []

# ================================
# Graceful shutdown
# ================================
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# ================================
# Run
# ================================
t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, daemon=True)

t1.start()
t2.start()
print(f"Ready to align your speech with the script ({len(script_lines)} lines). Press Ctrl+C to stop.")

t1.join()
t2.join()
print("Shutdown complete")
