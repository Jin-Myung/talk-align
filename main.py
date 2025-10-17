import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import re, signal, time, os, webbrowser
from typing import Dict, List
from wsbridge import WSBridge

# ================================
# Windows 콘솔 UTF-8 fix
# ================================
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
# Arguments
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
    cur_no, cur_buf = None, []

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
        elif cur_no is not None:
            cur_buf.append(ln)

    flush()
    return paras

# ================================
# Whisper + Audio + VAD
# ================================
model = WhisperModel("small", device="cpu", compute_type="int8")
RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
FORMAT = pyaudio.paInt16
vad = webrtcvad.Vad(1)

rt_audio_q = queue.Queue()
stop_event = threading.Event()
stream = None

def should_stop() -> bool:
    return stop_event.is_set()

# ================================
# Audio capture thread
# ================================
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
        pa.terminate()

# ================================
# Text utils
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

def clamp_idx(i: int) -> int:
    return max(0, min(TOTAL - 1, i))

# ================================
# Thresholds
# ================================
MIN_UTTER_TEXT_LEN_FOR_MATCH = 6
MIN_UTTER_LEN_IN_MS_FOR_MATCH = 1000
MAX_UTTER_LEN_IN_MS = 10000
SCRIPT_MATCH_THRESHOLD = 70

# ================================
# Process utterance (STT + alignment)
# ================================
def process_utterance(ws, utter, pending_text, pending_ms, cur_idx):
    audio = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(audio) / RATE * 1000)
    segments, _ = model.transcribe(audio, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if not raw_text:
        return pending_text, pending_ms, cur_idx

    combined_text = (pending_text + " " + raw_text).strip() if pending_text else raw_text
    combined_ms = pending_ms + utter_ms

    if significant_len(combined_text) < MIN_UTTER_TEXT_LEN_FOR_MATCH or combined_ms < MIN_UTTER_LEN_IN_MS_FOR_MATCH:
        ws.send({"type": "recognize", "text": combined_text})
        return combined_text, combined_ms, cur_idx

    ws.send({"type": "recognize", "text": combined_text})

    # 문단 매칭
    start = clamp_idx(cur_idx - 1)
    end = clamp_idx(cur_idx + 4)
    search_range = aligned_ko[start:end]

    if not search_range:
        return "", 0, cur_idx

    match, score, idx_rel = process.extractOne(
        combined_text, search_range, scorer=fuzz.partial_ratio
    )
    idx = start + idx_rel

    if score >= SCRIPT_MATCH_THRESHOLD:
        cur_idx = idx
        tag = aligned_tag[idx]
        en_line = aligned_en[idx]
        para_idx = aligned_para_idx[idx]

        # 문단 전체 전송
        cur_para = paras_en[para_idx]
        next_para = paras_en[para_idx + 1] if para_idx + 1 < len(paras_en) else ""
        ws.send({"type": "para_match", "para_idx": para_idx, "cur": cur_para, "next": next_para})
        print(f"[{tag}] {en_line} ({score:.1f}%)")
    else:
        ws.send({"type": "info", "msg": f"No match ({score:.1f}%)"})
        print(f"No matching script ({score:.1f}%)")

    return "", 0, cur_idx

# ================================
# VAD loop
# ================================
def vad_loop(ws):
    cur_idx = 0
    pending_text = ""
    pending_ms = 0
    in_utter = False
    utter = []
    speech_count = 0
    silence_count = 0
    speech_start_frames = 2
    silence_end_frames = 6

    # 첫 문단 전송
    ws.send({
        "type": "para_match",
        "para_idx": 0,
        "cur": paras_en[0],
        "next": paras_en[1] if len(paras_en) > 1 else ""
    })

    while not should_stop():
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
                    pending_text, pending_ms, cur_idx = process_utterance(ws, utter, pending_text, pending_ms, cur_idx)
                    in_utter = False
                    utter = []
                    speech_count = 0
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_text, pending_ms, cur_idx = process_utterance(ws, utter, pending_text, pending_ms, cur_idx)
                    in_utter = False
                    utter = []
                    speech_count = 0
                    silence_count = 0

# ================================
# Graceful shutdown
# ================================
def request_shutdown(*_):
    print("\nShutting down...")
    stop_event.set()
    try:
        if stream:
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
    print("Error: Start wsbridge server first (`uvicorn server:app --host 0.0.0.0 --port 8000`)")
    sys.exit(1)

operator_url = "http://127.0.0.1:8000/public/operator.html"
audience_url = "http://127.0.0.1:8000/public/audience.html"

print("Opening browser windows...")
webbrowser.open(operator_url)
webbrowser.open(audience_url)

time.sleep(2)

# --- Load and align paragraphs ---
ko_paras = parse_paragraphs(ko_file)
en_paras = parse_paragraphs(en_file)

if not ko_paras or not en_paras:
    print("Missing paragraphs in script files.")
    sys.exit(1)

para_ids = sorted(set(ko_paras.keys()) & set(en_paras.keys()))
if not para_ids:
    print("No common paragraph numbers between KO/EN files.")
    sys.exit(1)

aligned_ko, aligned_en, aligned_tag, aligned_para_idx = [], [], [], []
paras_ko, paras_en = [], []
for i, pid in enumerate(para_ids):
    ko_sent = " ".join(ko_paras[pid])
    en_sent = " ".join(en_paras[pid])
    paras_ko.append(ko_sent)
    paras_en.append(en_sent)
    for j, (k, e) in enumerate(zip(ko_paras[pid], en_paras[pid])):
        aligned_ko.append(k)
        aligned_en.append(e)
        aligned_tag.append(f"{pid}.{j+1}")
        aligned_para_idx.append(i)

TOTAL = len(aligned_ko)
print(f"Loaded {len(para_ids)} paragraphs ({TOTAL} aligned sentences)")

# 초기 문단 전체 전송 (operator UI 구성용)
ws.send({"type": "init_paras", "ko": paras_ko, "en": paras_en})

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
print("Shutdown complete.")
