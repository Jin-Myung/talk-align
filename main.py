import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import re
import signal
import time
from typing import Dict, List
import webbrowser
from wsbridge import WSBridge

# --- Windows console UTF-8 fix ---
import os
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ================================
# Args
# ================================
if len(sys.argv) < 3:
    print("Usage: python main.py <ko_script.txt> <en_prompt.txt>")
    sys.exit(1)

ko_file = sys.argv[1]
en_file = sys.argv[2]

# ================================
# Sentence splitter (language-agnostic)
# ================================
_sent_re = re.compile(r'([^.!?]*[.!?]["”\']?)')

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    parts = [m.group(0).strip() for m in _sent_re.finditer(text)]
    parts = [p for p in parts if p]
    tail = _sent_re.sub('', text).strip()
    if tail:
        parts.append(tail)
    return parts

# ================================
# Paragraph parser
# ================================
_para_start_re = re.compile(r'^\s*(\d+)\.\s*(.*)$')

def parse_paragraphs(path: str) -> Dict[int, List[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip() for ln in f]

    paras: Dict[int, List[str]] = {}
    cur_no: int = None
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_no, cur_buf
        if cur_no is None:
            return
        # skip if already exists (avoid duplication)
        if cur_no in paras:
            return
        paragraph_text = ' '.join([s for s in cur_buf if s.strip()])
        sents = [s.strip() for s in split_sentences(paragraph_text)]
        paras[cur_no] = [s for s in sents if s]
        cur_buf = []

    for ln in lines:
        m = _para_start_re.match(ln)
        if m:
            flush()
            cur_no = int(m.group(1))
            tail = m.group(2) or ""
            cur_buf = [tail] if tail.strip() else []
        else:
            if cur_no is not None:
                cur_buf.append(ln)

    flush()
    return paras

# ================================
# Initialize model / audio / VAD
# ================================
model = WhisperModel("small", device="cpu", compute_type="int8")

RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)

rt_audio_q = queue.Queue(maxsize=50)
stop_event = threading.Event()

def should_stop():
    return stop_event.is_set()

# ================================
# Thresholds
# ================================
MIN_UTTER_TEXT_LEN_FOR_MATCH = 6
MIN_UTTER_LEN_IN_MS_FOR_MATCH = 1000
MAX_UTTER_LEN_IN_MS = 10000
SCRIPT_MATCH_THRESHOLD = 70

# ================================
# Audio capture thread
# ================================
stream = None

def audio_capture():
    global stream
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while not should_stop():
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            rt_audio_q.put(frame)
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

# ================================
# Helper
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

def clamp_idx(i: int) -> int:
    return max(0, min(TOTAL - 1, i))

# ================================
# Process utterance
# ================================
def process_utterance(ws, utter, pending_text, pending_ms, cur_idx):
    normalized = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(normalized) / RATE * 1000)
    segments, _ = model.transcribe(normalized, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if not raw_text:
        return pending_text, pending_ms, cur_idx

    combined = (pending_text + " " + raw_text).strip() if pending_text else raw_text
    combined_ms = pending_ms + utter_ms

    if significant_len(combined) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and combined_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:
        print(f"\nRecognized: {combined}")
        ws.send({"type": "recognize", "text": combined})

        start = clamp_idx(cur_idx - 1)
        end = clamp_idx(cur_idx + 4)
        search_range = aligned_ko[start:end]

        if search_range:
            match, score, idx_rel = process.extractOne(combined, search_range, scorer=fuzz.partial_ratio)
            idx = start + idx_rel

            if score >= SCRIPT_MATCH_THRESHOLD:
                cur_idx = idx
                tag = aligned_tag[idx]
                en_line = aligned_en[idx]
                next_line = aligned_en[idx + 1] if idx + 1 < len(aligned_en) else ""
                print(f"[{tag}] EN: {en_line} (similarity {score:.3f}%)")
                ws.send({
                    "type": "match",
                    "idx": cur_idx,
                    "tag": tag,
                    "line": en_line,
                    "ko": aligned_ko[idx],
                    "score": round(score, 1),
                    "next": next_line,
                })
            else:
                print(f"No matching script (similarity {score:.3f}%)")
                ws.send({"type": "info", "msg": f"No match ({round(score,1)}%)"})
        else:
            ws.send({"type": "info", "msg": "No matching script (end of window)"})

        pending_text, pending_ms = "", 0
    else:
        pending_text, pending_ms = combined, combined_ms
        print(f"\nBuffering: chars={significant_len(pending_text)}, ms={pending_ms}")

    return pending_text, pending_ms, cur_idx

# ================================
# VAD + alignment loop
# ================================
def vad_loop(ws: WSBridge):
    cur_idx = 0
    utter = []
    in_utter = False
    pending_text = ""
    pending_ms = 0

    speech_start_frames = 2
    silence_end_frames = 6
    speech_count = 0
    silence_count = 0

    # Send initial paragraph set to operator
    ws.send({"type": "init_paras", "ko": aligned_ko_paras, "en": aligned_en_paras})

    if aligned_en:
        tag = aligned_tag[cur_idx]
        cur = aligned_en[cur_idx]
        nxt = aligned_en[cur_idx + 1] if cur_idx + 1 < len(aligned_en) else ""
        ws.send({"type": "para_match", "para_idx": cur_para_idx(cur_idx), "cur": cur, "next": nxt})

    while not should_stop():
        cmd = ws.get_cmd_nowait()
        if cmd:
            t = cmd.get("type")
            if t == "prev":
                cur_idx = max(0, cur_idx - para_step(cur_idx))
            elif t == "next":
                cur_idx = min(TOTAL - 1, cur_idx + para_step(cur_idx))

            # broadcast paragraph update
            ws.send({
                "type": "para_match",
                "para_idx": cur_para_idx(cur_idx),
                "cur": aligned_en_paras[cur_para_idx(cur_idx)],
                "next": aligned_en_paras[cur_para_idx(cur_idx) + 1]
                if cur_para_idx(cur_idx) + 1 < len(aligned_en_paras)
                else "",
            })

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
                    utter = [frame]
                    silence_count = 0
            else:
                speech_count = 0
        else:
            utter.append(frame)
            if is_speech:
                silence_count = 0
                if len(utter) * FRAME_DURATION >= MAX_UTTER_LEN_IN_MS:
                    pending_text, pending_ms, cur_idx = process_utterance(
                        ws, utter, pending_text, pending_ms, cur_idx
                    )
                    in_utter = False
                    speech_count = 0
                    utter = []
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_text, pending_ms, cur_idx = process_utterance(
                        ws, utter, pending_text, pending_ms, cur_idx
                    )
                    in_utter = False
                    speech_count = 0
                    silence_count = 0
                    utter = []

# ================================
# Helpers for paragraph tracking
# ================================
def cur_para_idx(line_idx: int) -> int:
    for p_i, (start, end) in enumerate(para_ranges):
        if start <= line_idx <= end:
            return p_i
    return 0

def para_step(line_idx: int) -> int:
    p_i = cur_para_idx(line_idx)
    start, end = para_ranges[p_i]
    return end - start + 1

# ================================
# Shutdown handling
# ================================
def request_shutdown(*_):
    print("\nShutting down...")
    stop_event.set()
    try:
        if stream is not None:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass
    try:
        ws.close()
    except Exception:
        pass

for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, request_shutdown)
    except Exception:
        pass
if hasattr(signal, "SIGBREAK"):
    try:
        signal.signal(signal.SIGBREAK, request_shutdown)
    except Exception:
        pass

# ================================
# Run
# ================================
try:
    ws = WSBridge("ws://127.0.0.1:8000/ws")
except ConnectionRefusedError:
    print("Error: Start wsbridge server first `uvicorn server:app --host 0.0.0.0 --port 8000`")
    sys.exit(1)

operator_url = "http://127.0.0.1:8000/public/operator.html"
audience_url = "http://127.0.0.1:8000/public/audience.html"

print("Opening URL for operator and audience:\n")
print(f"  - {operator_url}")
print(f"  - {audience_url}\n")
webbrowser.open(operator_url)
webbrowser.open(audience_url)

time.sleep(2)

print("Loading script and prompt files...")

# --- Parse paragraphs ---
ko_paras = parse_paragraphs(ko_file)
en_paras = parse_paragraphs(en_file)

if not ko_paras or not en_paras:
    print("Error: invalid KO/EN paragraph structure")
    sys.exit(1)

common_ids = sorted(set(ko_paras.keys()) & set(en_paras.keys()))
if not common_ids:
    print("No common paragraph numbers.")
    sys.exit(1)

aligned_ko, aligned_en, aligned_tag = [], [], []
aligned_ko_paras, aligned_en_paras, para_ranges = [], [], []
line_counter = 0

for pid in common_ids:
    ko_sents, en_sents = ko_paras[pid], en_paras[pid]
    n = min(len(ko_sents), len(en_sents))
    start = line_counter
    for i in range(n):
        aligned_ko.append(ko_sents[i])
        aligned_en.append(en_sents[i])
        aligned_tag.append(f"{pid}.{i+1}")
        line_counter += 1
    end = line_counter - 1
    para_ranges.append((start, end))
    aligned_ko_paras.append(" ".join(ko_sents))
    aligned_en_paras.append(" ".join(en_sents))

TOTAL = len(aligned_ko)
print(f"Loaded {len(common_ids)} paragraphs ({TOTAL} total sentences)")

t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, args=(ws,), daemon=True)

t1.start()
t2.start()
print("Ready. Press Ctrl+C to stop.")

try:
    while not should_stop():
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)
        if not t1.is_alive() and not t2.is_alive():
            break
except KeyboardInterrupt:
    request_shutdown()

for t in (t1, t2):
    t.join(timeout=2.0)
print("Shutdown complete")
