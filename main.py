import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)

from faster_whisper import WhisperModel
import gc
import io
import numpy as np
import os
import pyaudio
import queue
from rapidfuzz import fuzz, process
import re
import signal
import sys
import threading
import time
from typing import Dict, List
import webbrowser
import webrtcvad
from wsbridge import WSBridge


# Force utf-8 printing regardless of terminal code page in Windows
if os.name == "nt" and not os.environ.get("MSYSTEM"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Args
ko_file = None
en_file = None

if len(sys.argv) == 3:
    ko_file = sys.argv[1]
    en_file = sys.argv[2]
    print(f"Using provided script files: {ko_file}, {en_file}")
elif len(sys.argv) == 1:
    print("No input files provided. Waiting for upload via Operator UI...")
else:
    print("Usage: python main.py <ko_script.txt> <en_script.txt>")
    sys.exit(1)

# Sentence splitter (language-agnostic, simple)
# - splits on [.?!], keeps delimiters
# - collapses spaces/newlines
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

# Paragraph parser
# - Header lines before first "^\d+\." are ignored
# - Returns dict: {para_no: ["sent1", "sent2", ...]}
_para_start_re = re.compile(r'^\s*(\d+)\.\s*(.*)$')

def parse_paragraphs_from_text(text: str) -> Dict[int, List[str]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
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
        sents = [s for s in sents if s]

        def split_long_sentence(sent: str) -> List[str]:
            length = len(sent)
            if length < 100:
                return [sent]

            if length < 150:
                n = 2
            elif length < 200:
                n = 3
            else:
                n = 4

            chunks = []
            part_len = length // n
            start = 0
            for i in range(1, n):
                mid = part_len * i
                left = max(0, mid - 20)
                right = min(length, mid + 20)
                window = sent[left:right]

                # Candidates of split points: semicolon, comma, space
                cut = None
                semi_pos = window.find(';')
                comma_pos = window.find(',')
                space_pos = window.find(' ')

                if semi_pos != -1:
                    cut = left + semi_pos + 1
                elif comma_pos != -1:
                    cut = left + comma_pos + 1
                elif space_pos != -1:
                    cut = left + space_pos + 1
                else:
                    cut = mid

                chunks.append(sent[start:cut].strip())
                start = cut

            chunks.append(sent[start:].strip())

            # Merge too short chunks
            merged = []
            for chunk in chunks:
                if merged and len(chunk) < 20:
                    merged[-1] += " " + chunk
                else:
                    merged.append(chunk)
            return merged

        expanded_sents = []
        for s in sents:
            expanded_sents.extend(split_long_sentence(s))
        paras[cur_no] = expanded_sents
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

# Initialize model / audio / VAD
model = WhisperModel("small", device="cpu", compute_type="int8")

RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 480 samples
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)   # VAD sensitivity (0=permissive, 3=aggressive)

rt_audio_q = queue.Queue(maxsize=100)  # real-time audio frames

view_mode_in_para = False
is_muted = False

# Graceful shutdown event
stop_event = threading.Event()
def should_stop() -> bool:
    return stop_event.is_set()

# Thresholds
MIN_UTTER_TEXT_LEN_FOR_MATCH = 6      # require at least 6 significant chars
MIN_UTTER_LEN_IN_MS_FOR_MATCH = 1000  # and at least 1 sec of speech
MAX_UTTER_LEN_IN_MS = 10000  # truncate utterance to 10 sec in case of no silence
SCRIPT_MATCH_THRESHOLD = 70  # require at least 70% similarity

pa = None
stream = None

def audio_capture():
    global pa, stream
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=1, rate=RATE,
                     input=True, frames_per_buffer=FRAME_SIZE)
    try:
        while not should_stop():
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            if not is_muted:
                rt_audio_q.put(frame)
    finally:
        try:
            if stream.is_active():
                stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

# Keep letters/digits/KO when counting significant chars
_sig_re = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]")
def significant_len(s: str) -> int:
    return len(_sig_re.findall(s))

# Dynamic script loader
aligned_ko, aligned_en, aligned_tag, aligned_en_idx = [], [], [], []
aligned_ko_paras, aligned_en_paras, para_ranges = [], [], []
TOTAL = 0

def load_scripts_from_text(ws, ko_text: str, en_text: str):
    global aligned_ko, aligned_en, aligned_tag, aligned_en_idx
    global aligned_ko_paras, aligned_en_paras, para_ranges
    global TOTAL

    ko_paras = parse_paragraphs_from_text(ko_text)
    en_paras = parse_paragraphs_from_text(en_text)

    if not ko_paras or not en_paras:
        ws.send({"type": "info", "msg": "Invalid KO/EN paragraphs"})
        return

    common_ids = sorted(set(ko_paras.keys()) & set(en_paras.keys()))
    aligned_ko, aligned_en, aligned_tag, aligned_en_idx = [], [], [], []
    aligned_ko_paras, aligned_en_paras, para_ranges = [], [], []
    line_counter = 0

    for pid in common_ids:
        # Align sentences in each paragraph
        ko_sents, en_sents = ko_paras[pid], en_paras[pid]
        len_ko_sent = len(ko_sents)
        len_en_sent = len(en_sents)
        start = line_counter
        if len_ko_sent == len_en_sent:
            for i in range(len_ko_sent):
                aligned_tag.append(f"{pid}.{i+1}")
                aligned_en_idx.append(len(aligned_en))
                aligned_en.append(en_sents[i])
        elif len_ko_sent > len_en_sent:
            step_size = len_ko_sent // len_en_sent
            for i in range(len_ko_sent):
                aligned_tag.append(f"{pid}.{i+1}")
                aligned_en_idx.append(len(aligned_en) + min(len_en_sent-1, i//step_size))
            aligned_en.extend(en_sents)
        else:
            step_size = len_en_sent // len_ko_sent
            remainder = len_en_sent % len_ko_sent
            merged_en_sent = []
            merge_base = 0
            for i in range(len_ko_sent):
                take = step_size + (1 if remainder > 0 else 0)
                remainder -= 1 if remainder > 0 else 0
                merged_en_sent.append(" ".join(en_sents[merge_base:merge_base + take]))
                merge_base += take
                aligned_tag.append(f"{pid}.{i+1}")
                aligned_en_idx.append(len(aligned_en) + i)
            aligned_en.extend(merged_en_sent)

        aligned_ko.extend(ko_sents)
        line_counter += len_ko_sent
        para_ranges.append((start, line_counter - 1))
        aligned_ko_paras.append(" ".join(ko_sents))
        aligned_en_paras.append(" ".join(en_sents))

    TOTAL = len(aligned_ko)
    ws.send({
        "type": "update_texts",
        "ko": aligned_ko_paras,
        "en": aligned_en_paras,
        "aligned_ko": aligned_ko,
        "aligned_en": aligned_en,
        "aligned_en_idx": aligned_en_idx,
        "view_mode_in_para": view_mode_in_para
    })
    print(f"Loaded {len(common_ids)} paragraphs and {TOTAL} sentences")

# Process utterance: STT + script alignment
def process_utterance(ws, utter, pending_text, pending_ms, sent_idx):
    try:
        normalized = np.frombuffer(b"".join(utter), np.int16).astype(np.float32) / 32768.0
        utter_ms = int(len(normalized) / RATE * 1000)
        segments, _ = model.transcribe(normalized, language="ko", beam_size=1)
    except Exception as e:
        print("STT error:", e)
        ws.send({"type": "info", "msg": f"STT error: {e}"})
        return pending_text, pending_ms, sent_idx
    raw_text = "".join(seg.text for seg in segments).strip()

    if not raw_text:
        return pending_text, pending_ms, sent_idx

    combined = (pending_text + " " + raw_text).strip() if pending_text else raw_text
    combined_ms = pending_ms + utter_ms

    if significant_len(combined) >= MIN_UTTER_TEXT_LEN_FOR_MATCH and combined_ms >= MIN_UTTER_LEN_IN_MS_FOR_MATCH:
        ws.send({"type": "info", "msg": f"recognized: {combined}"})

        start, end = get_search_range(sent_idx)
        search_range = aligned_ko[start:end+1]

        if search_range:
            match, score, idx_rel = process.extractOne(combined, search_range, scorer=fuzz.partial_ratio)
            idx = start + idx_rel

            if score >= SCRIPT_MATCH_THRESHOLD:
                idx = min(idx + 1, TOTAL - 1) # highlight next sentence while recognizing current sentence
                sent_idx = idx
                tag = aligned_tag[idx]
                en_line = aligned_en[idx]
                print(f"[{tag}] EN: {en_line} (similarity {score:.3f}%)")
                ws.send({
                    "type": "match",
                    "script_idx": sent_idx,
                    "prompt_idx": aligned_en_idx[sent_idx],
                    "tag": tag,
                })
            else:
                print(f"No matching script (similarity {score:.3f}%)")
                ws.send({"type": "info", "msg": f"No match ({round(score,1)}%)"})
        else:
            ws.send({"type": "info", "msg": "No matching script (end of window)"})

        pending_text, pending_ms = "", 0
    else:
        pending_text, pending_ms = combined, combined_ms
        print(f"Buffering: chars={significant_len(pending_text)}, ms={pending_ms}")

    return pending_text, pending_ms, sent_idx

# Helpers for paragraph tracking
def sent_idx_to_para_idx(sent_idx: int) -> int:
    for para_idx, (start, end) in enumerate(para_ranges):
        if start <= sent_idx <= end:
            return para_idx
    raise IndexError("Sentence index out of range")

def para_idx_to_sent_idx(para_idx: int) -> tuple[int, int]:
    if para_idx < 0 or para_idx >= len(para_ranges):
        raise IndexError("Paragraph index out of range")
    else:
        start, end = para_ranges[para_idx]
        return start, end

def get_search_range(sent_idx: int) -> tuple[int, int]:
    if sent_idx < 0 or sent_idx >= TOTAL:
        raise IndexError("Sentence index out of range")
    para_idx = sent_idx_to_para_idx(sent_idx)
    start, end = para_idx_to_sent_idx(para_idx)

    # Expand search range by 1 paragraph before and 1 paragraphs after
    if para_idx > 0:
        prev_start, _ = para_idx_to_sent_idx(para_idx - 1)
        start = prev_start
    if para_idx < len(para_ranges) - 1:
        _, next_end = para_idx_to_sent_idx(para_idx + 1)
        end = next_end

    return start, end

# Handle file upload (Operator)
def handle_uploaded_files(ws, cmd):
    global view_mode_in_para

    ko_text = cmd.get("ko_text", "")
    en_text = cmd.get("en_text", "")
    view_mode_in_para = bool(cmd.get("view_mode_in_para", False))
    if not ko_text or not en_text:
        ws.send({"type": "info", "msg": "Both KO/EN text required"})
        return
    load_scripts_from_text(ws, ko_text, en_text)

cur_sent_idx = 0
cur_sent_idx_lock = threading.Lock()

# Command processing loop
def cmd_loop(ws: WSBridge):
    global view_mode_in_para, cur_sent_idx

    last_sent_idx = -1
    last_sent_para_idx = -1

    if aligned_ko and para_ranges:
        ws.send({
            "type": "move",
            "para_idx": sent_idx_to_para_idx(0),
            "sent_idx_in_para": 0,
            "script_idx": 0,
            "prompt_idx": aligned_en_idx[0] if aligned_en_idx else 0,
            "tag": aligned_tag[0] if aligned_tag else "",
            "reloading": True
        })

    while not should_stop():
        try:
            cmd = ws.get_cmd(timeout=0.1)
        except queue.Empty:
            continue
        except Exception:
            time.sleep(0.05)
            continue

        if not cmd:
            continue

        t = ""
        if cmd:
            t = cmd.get("type")
            if t == "prev":
                with cur_sent_idx_lock:
                    if view_mode_in_para:
                        para_idx = sent_idx_to_para_idx(cur_sent_idx)
                        if para_idx > 0:
                            cur_sent_idx, _ = para_idx_to_sent_idx(para_idx - 1)
                    else:
                        if cur_sent_idx > 0:
                            cur_sent_idx -= 1
            elif t == "next":
                with cur_sent_idx_lock:
                    if view_mode_in_para:
                        para_idx = sent_idx_to_para_idx(cur_sent_idx)
                        if para_idx < len(para_ranges) - 1:
                            cur_sent_idx, _ = para_idx_to_sent_idx(para_idx + 1)
                    else:
                        if cur_sent_idx < TOTAL - 1:
                            cur_sent_idx += 1
            elif t == "go_to":
                with cur_sent_idx_lock:
                    if view_mode_in_para:
                        para_idx = cmd.get("idx", 0)
                        if para_idx >= 0 and para_idx < len(para_ranges):
                            cur_sent_idx, _ = para_idx_to_sent_idx(para_idx)
                    else:
                        script_idx = cmd.get("idx", 0)
                        if script_idx >= 0 and script_idx < TOTAL:
                            cur_sent_idx = script_idx
            elif t == "load_files":
                with cur_sent_idx_lock:
                    handle_uploaded_files(ws, cmd)
                    cur_sent_idx = 0
                    last_sent_para_idx = -1
            elif t == "mute":
                global is_muted
                is_muted = bool(cmd.get("value", False))
                print(f"Mute {'ON' if is_muted else 'OFF'}")
            elif t == "change_view_mode":
                with cur_sent_idx_lock:
                    view_mode_in_para = bool(cmd.get("view_mode_in_para", False))
                    cur_sent_idx = 0
                    last_sent_para_idx = -1
                    print(f"View in {'paragraph' if view_mode_in_para else 'sentence'}")
                    ws.send({
                        "type": "update_texts",
                        "ko": aligned_ko_paras,
                        "en": aligned_en_paras,
                        "aligned_ko": aligned_ko,
                        "aligned_en": aligned_en,
                        "aligned_en_idx": aligned_en_idx,
                        "view_mode_in_para": view_mode_in_para
                    })

        if not aligned_ko:
            time.sleep(0.1)
            continue

        with cur_sent_idx_lock:
            para_idx = sent_idx_to_para_idx(cur_sent_idx)
            base_sent_idx, _ = para_idx_to_sent_idx(para_idx)
            if para_idx != last_sent_para_idx or t == "change_view_mode" or t == "load_files":
                last_sent_para_idx = para_idx
                last_sent_idx = cur_sent_idx
                ws.send({
                    "type": "move",
                    "para_idx": para_idx,
                    "sent_idx_in_para": cur_sent_idx - base_sent_idx,
                    "script_idx": cur_sent_idx,
                    "prompt_idx": aligned_en_idx[cur_sent_idx],
                    "tag": aligned_tag[cur_sent_idx],
                    "reloading": t == "change_view_mode" or t == "load_files",
                })
            elif cur_sent_idx != last_sent_idx:
                last_sent_idx = cur_sent_idx
                ws.send({
                    "type": "move",
                    "para_idx": para_idx,
                    "sent_idx_in_para": cur_sent_idx - base_sent_idx,
                    "script_idx": cur_sent_idx,
                    "prompt_idx": aligned_en_idx[cur_sent_idx],
                    "tag": aligned_tag[cur_sent_idx],
                    "reloading": False,
                })

# VAD + alignment loop
def vad_loop(ws: WSBridge):
    global cur_sent_idx

    utter = []
    in_utter = False
    pending_text = ""
    pending_ms = 0

    speech_start_frames = 2  # 2 * FRAME_DURATION = 60ms to confirm speech start
    silence_end_frames = 4   # 4 * FRAME_DURATION = 120ms to confirm end
    speech_count = 0
    silence_count = 0

    while not should_stop():
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
                    # force process when utterance is too long
                    with cur_sent_idx_lock:
                        pending_text, pending_ms, cur_sent_idx = process_utterance(
                            ws, utter, pending_text, pending_ms, cur_sent_idx
                        )
                    in_utter = False
                    speech_count = 0
                    utter = []
            else:
                silence_count += 1
                if silence_count >= silence_end_frames:
                    with cur_sent_idx_lock:
                        pending_text, pending_ms, cur_sent_idx = process_utterance(
                            ws, utter, pending_text, pending_ms, cur_sent_idx
                        )
                    in_utter = False
                    speech_count = 0
                    silence_count = 0
                    utter = []

def heartbeat_loop(ws):
    interval = 1  # seconds
    while not should_stop():
        try:
            ws.send({"type": "alive", "ts": time.time()})
        except Exception:
            pass
        time.sleep(interval)

# Shutdown handling
def request_shutdown(*_):
    print("\nShutting down...")
    stop_event.set()
    try:
        rt_audio_q.put_nowait(b"")
    except Exception:
        pass
    try:
        if stream is not None:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass
    try:
        if pa is not None:
            pa.terminate()
    except Exception:
        pass
    try:
        ws.close()
    except Exception:
        pass
    try:
        del model
        gc.collect()
    except Exception:
        pass
    # wait a bit to let native threads exit cleanly
    time.sleep(0.3)

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
    print("Error: Start webserver first `uvicorn server:app --host 0.0.0.0 --port 8000`")
    sys.exit(1)
except Exception as e:
    print("WebSocket connection error: ", e)
    sys.exit(1)

operator_url = "http://127.0.0.1:8000/public/operator.html"
audience_url = "http://127.0.0.1:8000/public/audience.html"

print("Opening URL for operator and audience:\n")
print(f"  - {operator_url}")
print(f"  - {audience_url}\n")

time.sleep(2)  # wait for WS to stabilize

if ko_file and en_file:
    print("Loading script and prompt files...")
    with open(ko_file, encoding="utf-8") as f1, open(en_file, encoding="utf-8") as f2:
        load_scripts_from_text(ws, f1.read(), f2.read())
else:
    print("No initial files — will wait for Operator to upload.")
    ws.send({"type": "update_texts", "ko": [], "en": [], "aligned_en": [], "view_mode_in_para": view_mode_in_para})

t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, args=(ws,), daemon=True)
t3 = threading.Thread(target=heartbeat_loop, args=(ws,), daemon=True)
t4 = threading.Thread(target=cmd_loop, args=(ws,), daemon=True)

t1.start()
t2.start()
t3.start()
t4.start()
print("Ready. Press Ctrl+C to stop.")

try:
    while not should_stop():
        time.sleep(1)
except KeyboardInterrupt:
    request_shutdown()

for t in (t1, t2, t3, t4):
    t.join(timeout=1.0)
time.sleep(0.5)  # ensure background threads fully shutdown
gc.collect()
print("Shutdown complete")
