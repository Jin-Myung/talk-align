import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import re
import signal
import time
from typing import Dict, List
import webbrowser
from wsbridge import WSBridge

# ================================
# Args
# ================================
if len(sys.argv) < 3:
    print("Usage: python main.py <ko_script.txt> <en_prompt.txt>")
    sys.exit(1)

ko_file = sys.argv[1]
en_file = sys.argv[2]

# ================================
# Sentence splitter (language-agnostic, simple)
# - splits on [.?!], keeps delimiters
# - collapses spaces/newlines
# ================================
_sent_re = re.compile(r'([^.!?]*[.!?]["”\']?)')

def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text.strip())
    if not text:
        return []
    parts = [m.group(0).strip() for m in _sent_re.finditer(text)]
    parts = [p for p in parts if p]  # remove empties
    # If tail has no terminal punctuation, include it as a final sentence
    tail = _sent_re.sub('', text).strip()
    if tail:
        parts.append(tail)
    return parts

# ================================
# Paragraph parser
# - Header lines before first "^\d+\." are ignored
# - Returns dict: {para_no: ["sent1", "sent2", ...]}
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
        if cur_no <= len(paras):
            return
        paragraph_text = ' '.join([s for s in cur_buf if s.strip()])
        sents = [s.strip() for s in split_sentences(paragraph_text)]
        paras[cur_no] = [s for s in sents if s]
        cur_buf = []

    for ln in lines:
        m = _para_start_re.match(ln)
        if m:
            # new paragraph
            flush()
            cur_no = int(m.group(1))
            tail = m.group(2) or ""
            cur_buf = [tail] if tail.strip() else []
        else:
            # header or paragraph body continuation
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

vad = webrtcvad.Vad(1)   # VAD sensitivity (0=permissive, 3=aggressive)

rt_audio_q = queue.Queue()

# ---- graceful shutdown event ----
stop_event = threading.Event()
def should_stop() -> bool:
    return stop_event.is_set()

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
stream = None

def audio_capture():
    global stream
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while not should_stop():
            # in case of overflow, skip frame
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
# Keep letters/digits/Hangul when counting significant chars
# ================================
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

# ================================
# Utils
# ================================
def clamp_idx(i: int) -> int:
    return max(0, min(TOTAL - 1, i))

# ================================
# Process utterance: STT + script alignment
# ================================
def process_utterance(ws, utter, pending_utter_text, pending_utter_len, current_line_in_script):
    normalized_utter = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
    utter_ms = int(len(normalized_utter) / RATE * 1000)

    segments, _ = model.transcribe(normalized_utter, language="ko", beam_size=1)
    raw_text = "".join(seg.text for seg in segments).strip()

    if raw_text:
        combined_text = (pending_utter_text + " " + raw_text).strip() if pending_utter_text else raw_text
        combined_ms = pending_utter_len + utter_ms

        if significant_len(combined_text) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and combined_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:
            print(f"\nRecognized: {combined_text}")
            ws.send({"type": "recognize", "text": combined_text})

            start = clamp_idx(current_line_in_script - 1)
            end = clamp_idx(current_line_in_script + 4)
            search_range = aligned_ko[start:end]

            if search_range:
                match, score, idx_rel = process.extractOne(
                    combined_text, search_range, scorer=fuzz.partial_ratio
                )
                idx = start + idx_rel

                if score >= SCRIPT_MATCH_THRESHOLD:
                    current_line_in_script = idx
                    tag = aligned_tag[idx]
                    en_line = aligned_en[idx]
                    next_line = aligned_en[idx+1] if idx+1 < len(aligned_en) else ""
                    print(f"[{tag}] EN: {en_line} (similarity {score:.3f}%)")
                    ws.send({"type": "match", "idx": idx, "tag": tag, "line": en_line, "ko": aligned_ko[idx], "score": round(score, 1), "next": next_line})
                else:
                    print(f"No matching script (similarity {score:.3f}%)")
                    ws.send({"type":"info", "msg": f"No match ({round(score,1)}%)"})
            else:
                print("No matching script (end of script window)")
                ws.send({"type":"info", "msg":"No matching script (window end)"})

            pending_utter_text = ""
            pending_utter_len = 0
        else:
            pending_utter_text = combined_text
            pending_utter_len = combined_ms
            print(f"\nBuffering: "
                  f"chars={significant_len(pending_utter_text)}/{MIN_UTTER_TEXT_LEN_FOR_MATCH}, "
                  f"ms={pending_utter_len}/{MIN_UTTER_LEN_IN_MS_FOR_MATCH}")

    return pending_utter_text, pending_utter_len, current_line_in_script

# ================================
# VAD + utterance processing loop
# ================================
def vad_loop(ws: WSBridge):
    current_line_in_script = 0
    utter = []
    in_utter = False

    # If an utterance is too short, we buffer it and prepend to the next one rather than matching a script line.
    pending_utter_text = ""
    pending_utter_len = 0  # in ms

    # VAD hangover settings
    speech_start_frames = 2  # 2 * FRAME_DURATION = 60ms to confirm speech start
    silence_end_frames = 6   # 6 * FRAME_DURATION = 180ms to confirm end
    speech_count = 0
    silence_count = 0

    if aligned_en:
        tag = aligned_tag[current_line_in_script]
        cur = aligned_en[current_line_in_script]
        nxt = aligned_en[current_line_in_script + 1] if current_line_in_script + 1 < len(aligned_en) else ""
        ws.send({"type": "match", "idx": current_line_in_script, "tag": tag, "line": cur, "ko": aligned_ko[current_line_in_script], "score": "", "next": nxt})
    else:
        ws.send({"type": "info", "msg": "Script is empty."})

    while not should_stop():
        # commands from operator UI
        cmd = ws.get_cmd_nowait()
        if cmd:
            t = cmd.get("type")
            if t == "prev":
                current_line_in_script = clamp_idx(current_line_in_script - 1)
                tag = aligned_tag[current_line_in_script]
                cur_en = aligned_en[current_line_in_script]
                nxt_en = aligned_en[current_line_in_script + 1] if current_line_in_script + 1 < len(aligned_en) else ""
                print(f"[OP] prev -> {tag}")
                ws.send({"type": "match", "idx": current_line_in_script, "tag": tag, "line": cur_en, "ko": aligned_ko[current_line_in_script], "score": "", "next": nxt_en})
            elif t == "next":
                current_line_in_script = clamp_idx(current_line_in_script + 1)
                tag = aligned_tag[current_line_in_script]
                cur_en = aligned_en[current_line_in_script]
                nxt_en = aligned_en[current_line_in_script + 1] if current_line_in_script + 1 < len(aligned_en) else ""
                print(f"[OP] next -> {tag}")
                ws.send({"type": "match", "idx": current_line_in_script, "tag": tag, "line": cur_en, "ko": aligned_ko[current_line_in_script], "score": "", "next": nxt_en})

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
                    # Force process if utterance too long
                    pending_utter_text, pending_utter_len, current_line_in_script = process_utterance(
                        ws, utter, pending_utter_text, pending_utter_len, current_line_in_script
                    )
                    # reset for next utterance
                    in_utter = False
                    speech_count = 0
                    utter = []
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    pending_utter_text, pending_utter_len, current_line_in_script = process_utterance(
                        ws, utter, pending_utter_text, pending_utter_len, current_line_in_script
                    )
                    # reset for next utterance
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
except ConnectionRefusedError:
    print("Error: Start wsbridge server first `uvicorn server:app --host 0.0.0.0 --port 8000`")
    sys.exit(1)
except Exception as e:
    print("WebSocket connection error: ", e)
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

aligned_ko: List[str] = []
aligned_en: List[str] = []
aligned_tag: List[str] = []  # e.g., "1.2"

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
