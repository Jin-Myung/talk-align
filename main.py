import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import re, signal, time, os, webbrowser
from typing import Dict, List
from wsbridge import WSBridge

# --- Windows console UTF-8 fix ---
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
# Sentence splitter
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
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)

rt_audio_q = queue.Queue()
stop_event = threading.Event()
def should_stop(): return stop_event.is_set()

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
            stream.stop_stream(); stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

# ================================
# Helpers
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str): return len(_sig_re.findall(s))
def clamp_idx(i: int): return max(0, min(TOTAL - 1, i))

# ================================
# Process utterance
# ================================
def process_utterance(ws, utter, pending_text, pending_ms, current_idx):
    normalized = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(normalized) / RATE * 1000)
    segments, _ = model.transcribe(normalized, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if raw_text:
        combined = (pending_text + " " + raw_text).strip() if pending_text else raw_text
        total_ms = pending_ms + utter_ms

        if significant_len(combined) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and total_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:
            print(f"\nRecognized: {combined}")
            ws.send({"type": "recognize", "text": combined})

            start = clamp_idx(current_idx - 1)
            end = clamp_idx(current_idx + 4)
            search_range = aligned_ko[start:end]

            if search_range:
                match, score, idx_rel = process.extractOne(
                    combined, search_range, scorer=fuzz.partial_ratio)
                idx = start + idx_rel

                if score >= SCRIPT_MATCH_THRESHOLD:
                    current_idx = idx
                    tag = aligned_tag[idx]
                    para_id = int(tag.split(".")[0])

                    # 문단 내 문장 줄바꿈 없이 결합
                    cur_para = " ".join(para_en.get(para_id, []))
                    next_para = ""
                    if para_id in para_order:
                        i = para_order.index(para_id)
                        if i + 1 < len(para_order):
                            next_para = " ".join(para_en.get(para_order[i + 1], []))

                    print(f"[{tag}] Paragraph {para_id} matched.")
                    ws.send({
                        "type": "para_match",
                        "para_id": para_id,
                        "cur": cur_para,
                        "next": next_para,
                        "ko": aligned_ko[idx],
                        "score": round(score, 1)
                    })
                else:
                    ws.send({"type":"info", "msg": f"No match ({round(score,1)}%)"})
            else:
                ws.send({"type":"info", "msg":"No matching script (window end)"})

            pending_text, pending_ms = "", 0
        else:
            pending_text, pending_ms = combined, total_ms
            print(f"Buffering ({significant_len(pending_text)}/{MIN_UTTER_TEXT_LEN_FOR_MATCH} chars)")
    return pending_text, pending_ms, current_idx

# ================================
# VAD loop
# ================================
def vad_loop(ws: WSBridge):
    cur_idx = 0
    utter = []
    in_utter = False
    pending_text, pending_ms = "", 0
    speech_start_frames, silence_end_frames = 2, 6
    speech_count = silence_count = 0

    # initial display
    if para_order:
        pid = para_order[0]
        nxt = para_order[1] if len(para_order) > 1 else None
        ws.send({
            "type": "para_match",
            "para_id": pid,
            "cur": " ".join(para_en.get(pid, [])),
            "next": " ".join(para_en.get(nxt, [])) if nxt else ""
        })
    else:
        ws.send({"type": "info", "msg": "Script empty."})

    while not should_stop():
        try:
            frame = rt_audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if should_stop(): break
        if len(frame) < FRAME_SIZE * 2: continue

        is_speech = vad.is_speech(frame, RATE)
        if not in_utter:
            if is_speech:
                speech_count += 1
                if speech_count >= speech_start_frames:
                    in_utter, utter = True, [frame]
                    silence_count = 0
            else:
                speech_count = 0
        else:
            utter.append(frame)
            if is_speech:
                silence_count = 0
                if len(utter) * FRAME_DURATION >= MAX_UTTER_LEN_IN_MS:
                    pending_text, pending_ms, cur_idx = process_utterance(ws, utter, pending_text, pending_ms, cur_idx)
                    in_utter, speech_count, utter = False, 0, []
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_text, pending_ms, cur_idx = process_utterance(ws, utter, pending_text, pending_ms, cur_idx)
                    in_utter, speech_count, silence_count, utter = False, 0, 0, []

# ================================
# Shutdown
# ================================
def request_shutdown(*_):
    print("\nShutting down...")
    stop_event.set()
    try:
        if stream: stream.stop_stream(); stream.close()
    except Exception:
        pass
    try: ws.close()
    except Exception:
        pass

for sig in (signal.SIGINT, signal.SIGTERM):
    try: signal.signal(sig, request_shutdown)
    except Exception: pass

# ================================
# Run
# ================================
try:
    ws = WSBridge("ws://127.0.0.1:8000/ws")
except Exception as e:
    print("WebSocket connection error:", e)
    sys.exit(1)

operator_url = "http://127.0.0.1:8000/public/operator.html"
audience_url = "http://127.0.0.1:8000/public/audience.html"
print(f"\nOpening URLs:\n  {operator_url}\n  {audience_url}\n")
webbrowser.open(operator_url); webbrowser.open(audience_url)
time.sleep(2)

print("Loading script and prompt files...")
ko_paras = parse_paragraphs(ko_file)
en_paras = parse_paragraphs(en_file)
if not ko_paras or not en_paras:
    print("Empty script files.")
    sys.exit(1)

para_ids = sorted(set(ko_paras.keys()) & set(en_paras.keys()))
if not para_ids:
    print("No common paragraphs.")
    sys.exit(1)

aligned_ko, aligned_en, aligned_tag = [], [], []
for p in para_ids:
    ko_s, en_s = ko_paras[p], en_paras[p]
    n = min(len(ko_s), len(en_s))
    for i in range(n):
        aligned_ko.append(ko_s[i])
        aligned_en.append(en_s[i])
        aligned_tag.append(f"{p}.{i+1}")

TOTAL = len(aligned_ko)
print(f"Loaded {TOTAL} sentences in {len(para_ids)} paragraphs.")

# build paragraph dictionary for audience
para_en = {p: en_paras[p] for p in para_ids}
para_order = sorted(para_en.keys())

t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, args=(ws,), daemon=True)
t1.start(); t2.start()

print("Ready. Press Ctrl+C to stop.")
try:
    while not should_stop():
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)
        if not t1.is_alive() and not t2.is_alive():
            break
except KeyboardInterrupt:
    request_shutdown()

t1.join(timeout=1)
t2.join(timeout=1)
print("Shutdown complete.")
