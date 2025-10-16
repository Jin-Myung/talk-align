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
#  - header ignored until first "^\d+\."
#  - stop parsing on duplicated paragraph id (handles KO+EN concatenated files)
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
        paragraph_text = ' '.join([s for s in cur_buf if s.strip()])
        sents = [s.strip() for s in split_sentences(paragraph_text)]
        paras[cur_no] = [s for s in sents if s]
        cur_buf = []

    for ln in lines:
        m = _para_start_re.match(ln)
        if m:
            p = int(m.group(1))
            # 이미 같은 문단 번호가 등장했다면, 이후는 EN 프롬프트로 간주하고 파싱 중단
            if p in paras:
                break
            # 이전 문단 마감 후 새 문단 시작
            flush()
            cur_no = p
            tail = (m.group(2) or "").strip()
            cur_buf = [tail] if tail else []
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
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 480 samples
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)   # 0..3

rt_audio_q = queue.Queue()

# ---- graceful shutdown event ----
stop_event = threading.Event()
def should_stop() -> bool:
    return stop_event.is_set()

# ================================
# Utterance thresholds
# ================================
MIN_UTTER_TEXT_LEN_FOR_MATCH = 6
MIN_UTTER_LEN_IN_MS_FOR_MATCH = 1000
MAX_UTTER_LEN_IN_MS = 10000
SCRIPT_MATCH_THRESHOLD = 70

# ================================
# Keep letters/digits/Hangul when counting significant chars
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

# ================================
# Globals filled after parsing
# ================================
aligned_ko: List[str] = []
aligned_en: List[str] = []
aligned_tag: List[str] = []
TOTAL = 0

def clamp_idx(i: int) -> int:
    return max(0, min(TOTAL - 1, i))

# ================================
# Audio capture thread
# ================================
pa = None
stream = None

def audio_capture():
    global pa, stream
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while not should_stop():
            # overflow 시 예외 대신 프레임 스킵
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
# Process utterance: STT(KO) + alignment → EN
# ================================
def process_utterance(ws, utter, pending_utter_text, pending_utter_len, current_idx):
    wav = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(wav) / RATE * 1000)

    segments, _ = model.transcribe(wav, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if raw_text:
        combined_text = (pending_utter_text + " " + raw_text).strip() if pending_utter_text else raw_text
        combined_ms = pending_utter_len + utter_ms

        if significant_len(combined_text) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and combined_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:
            print(f"\nRecognized: {combined_text}")
            ws.send({"type": "recognize", "text": combined_text})

            start = clamp_idx(current_idx - 1)
            end   = clamp_idx(current_idx + 4)
            cand_indices = list(range(start, end + 1))
            if current_idx in cand_indices and len(cand_indices) > 1:
                cand_indices.remove(current_idx)

            if cand_indices:
                cand_texts = [aligned_ko[i] for i in cand_indices]
                match, score, idx_rel = process.extractOne(combined_text, cand_texts, scorer=fuzz.partial_ratio)
                idx = cand_indices[idx_rel]

                if score >= SCRIPT_MATCH_THRESHOLD:
                    current_idx = idx
                    tag = aligned_tag[idx]
                    en_line = aligned_en[idx]
                    next_line = aligned_en[idx+1] if idx+1 < TOTAL else ""
                    print(f"[{tag}] EN: {en_line} (similarity {score:.3f}%)")
                    ws.send({
                        "type": "match",
                        "idx": idx,
                        "tag": tag,
                        "line": en_line,
                        "ko": aligned_ko[idx],
                        "score": round(score, 1),
                        "next": next_line
                    })
                else:
                    print(f"No matching script (similarity {score:.3f}%)")
                    ws.send({"type":"info", "msg": f"No match ({round(score,1)}%)"})
            else:
                print("No matching script (window empty)")
                ws.send({"type":"info", "msg":"No matching script (window empty)"})

            pending_utter_text = ""
            pending_utter_len = 0
        else:
            pending_utter_text = combined_text
            pending_utter_len = combined_ms
            print(f"\nBuffering: chars={significant_len(pending_utter_text)}/{MIN_UTTER_TEXT_LEN_FOR_MATCH}, "
                  f"ms={pending_utter_len}/{MIN_UTTER_LEN_IN_MS_FOR_MATCH}")

    return pending_utter_text, pending_utter_len, current_idx

# ================================
# VAD + utterance processing loop
# ================================
def vad_loop(ws: WSBridge):
    current_idx = 0
    utter = []
    in_utter = False

    pending_utter_text = ""
    pending_utter_len = 0  # ms

    # VAD hangover
    speech_start_frames = 2   # 60ms
    silence_end_frames = 6    # 180ms
    speech_count = 0
    silence_count = 0

    # Startup show first EN line
    if TOTAL > 0:
        tag = aligned_tag[current_idx]
        cur = aligned_en[current_idx]
        nxt = aligned_en[current_idx + 1] if current_idx + 1 < TOTAL else ""
        ws.send({"type": "match", "idx": current_idx, "tag": tag, "line": cur,
                 "ko": aligned_ko[current_idx], "score": "", "next": nxt})
    else:
        ws.send({"type": "info", "msg": "Script is empty."})

    while not should_stop():
        # operator commands
        cmd = ws.get_cmd_nowait()
        if cmd:
            t = cmd.get("type")
            if t == "prev":
                current_idx = clamp_idx(current_idx - 1)
                tag = aligned_tag[current_idx]
                cur_en = aligned_en[current_idx]
                nxt_en = aligned_en[current_idx + 1] if current_idx + 1 < TOTAL else ""
                print(f"[OP] prev -> {tag}")
                ws.send({"type": "match", "idx": current_idx, "tag": tag, "line": cur_en,
                         "ko": aligned_ko[current_idx], "score": "", "next": nxt_en})
            elif t == "next":
                current_idx = clamp_idx(current_idx + 1)
                tag = aligned_tag[current_idx]
                cur_en = aligned_en[current_idx]
                nxt_en = aligned_en[current_idx + 1] if current_idx + 1 < TOTAL else ""
                print(f"[OP] next -> {tag}")
                ws.send({"type": "match", "idx": current_idx, "tag": tag, "line": cur_en,
                         "ko": aligned_ko[current_idx], "score": "", "next": nxt_en})

        try:
            frame = rt_audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if should_stop():
            break

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
                    pending_utter_text, pending_utter_len, current_idx = process_utterance(
                        ws, utter, pending_utter_text, pending_utter_len, current_idx
                    )
                    in_utter = False
                    speech_count = 0
                    utter = []
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_utter_text, pending_utter_len, current_idx = process_utterance(
                        ws, utter, pending_utter_text, pending_utter_len, current_idx
                    )
                    in_utter = False
                    speech_count = 0
                    silence_count = 0
                    utter = []

# ================================
# Signal handling / graceful shutdown
# ================================
def request_shutdown(*_):
    print("\nShutting down...")
    stop_event.set()
    # 깨워주기
    try:
        if stream is not None:
            stream.stop_stream(); stream.close()
    except Exception:
        pass
    try:
        rt_audio_q.put_nowait(b"")
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
except Exception as e:
    print("WebSocket connection error: ", e)
    print("Start the server first: uvicorn server:app --host 127.0.0.1 --port 8000")
    sys.exit(1)

operator_url = "http://127.0.0.1:8000/public/operator.html"
audience_url = "http://127.0.0.1:8000/public/audience.html"

print("Opening URL for operator and audience:\n")
print(f"  - {operator_url}")
print(f"  - {audience_url}\n")

webbrowser.open(operator_url)
webbrowser.open(audience_url)

time.sleep(2)  # wait for WS to stabilize

print("Loading script and prompt files:")

# Load KO & EN paragraphs
ko_paras = parse_paragraphs(ko_file)
en_paras = parse_paragraphs(en_file)

if not ko_paras:
    print("Korean script has no paragraphs starting with '<n>.'")
    sys.exit(1)
if not en_paras:
    print("English prompt has no paragraphs starting with '<n>.'")
    sys.exit(1)

# Build flat alignment list by paragraph intersection
para_ids = sorted(set(ko_paras.keys()) & set(en_paras.keys()))
if not para_ids:
    print("No common paragraph numbers between KO/EN files.")
    sys.exit(1)

for p in para_ids:
    ko_sents = ko_paras[p]
    en_sents = en_paras[p]
    if len(ko_sents) != len(en_sents):
        print(f"Warning: sentence count mismatch in paragraph {p}: KO={len(ko_sents)} vs EN={len(en_sents)}")
    n = min(len(ko_sents), len(en_sents))
    for i in range(n):
        aligned_ko.append(ko_sents[i])
        aligned_en.append(en_sents[i])
        aligned_tag.append(f"{p}.{i+1}")

if not aligned_ko:
    print("No aligned sentences after parsing.")
    sys.exit(1)

TOTAL = len(aligned_ko)
print(f"Loaded alignment: {TOTAL} sentences across {len(para_ids)} paragraphs")

t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, args=(ws,), daemon=True)

t1.start()
t2.start()
print(f"Ready. KO={ko_file}, EN={en_file}. Total aligned sentences: {TOTAL}. Press Ctrl+C to stop.")

try:
    while not should_stop():
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)
        if not t1.is_alive() and not t2.is_alive():
            break
except KeyboardInterrupt:
    request_shutdown()

# final join
for t in (t1, t2):
    t.join(timeout=2.0)
print("Shutdown complete")
