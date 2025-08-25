import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz
import signal

# ================================
# 1. 대본 읽기
# ================================
if len(sys.argv) < 2:
    print("사용법: python vad_align.py <script.txt>")
    sys.exit(1)

script_file = sys.argv[1]
with open(script_file, "r", encoding="utf-8") as f:
    script_lines = [line.strip() for line in f if line.strip()]

print(f"📖 대본 로드 완료 ({len(script_lines)} 줄)")

# ================================
# 2. 모델/오디오/VAD 초기화
# ================================
model = WhisperModel("small", device="cpu", compute_type="int8")

RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # 480 samples
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)   # 민감도 설정 (0=느슨, 3=엄격)

audio_q = queue.Queue()
running = True
current_index = 0  # 대본 진행 위치

# ================================
# 3. 오디오 캡처 스레드
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
# 4. VAD + STT + 대본 매칭
# ================================
def vad_loop():
    global current_index
    buffer = []
    is_speaking = False
    threshold = 70  # 유사도 임계치

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
                # 발화 종료 → STT 실행
                audio = np.frombuffer(b"".join(buffer), np.int16).astype(np.float32) / 32768.0
                segments, _ = model.transcribe(audio, language="ko", beam_size=1)
                text = "".join(seg.text for seg in segments).strip()

                if text:
                    # ================================
                    # 대본과 비교 (현재 라인부터 앞으로 3줄까지만)
                    # ================================
                    start = current_index
                    end = min(len(script_lines), current_index + 4)
                    search_range = script_lines[start:end]

                    match, score, idx_rel = process.extractOne(
                        text, search_range, scorer=fuzz.partial_ratio
                    )
                    idx = start + idx_rel

                    if score >= threshold:
                        current_index = idx
                        print(f"\n📝 인식: {text}")
                        print(f"➡️ 대본[{idx}]: {match} (유사도 {score}%)")
                    else:
                        print(f"\n📝 인식: {text}")
                        print(f"❓ 대본과 매칭 실패 (유사도 {score}%)")

                buffer = []
                is_speaking = False

    print("🛑 VAD loop stopped")

# ================================
# 5. Graceful Shutdown
# ================================
def signal_handler(sig, frame):
    global running
    print("\n🛑 종료 중...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# ================================
# 6. 실행
# ================================
t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, daemon=True)

t1.start()
t2.start()
t1.join()
t2.join()
print("✅ 종료 완료")
