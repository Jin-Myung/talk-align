import sys, pyaudio, numpy as np, queue, threading, webrtcvad
from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz

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
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
FORMAT = pyaudio.paInt16

vad = webrtcvad.Vad(1)  # 민감도 설정 (0-3): 3이 가장 민감

audio_q = queue.Queue()
running = True
current_index = 0  # 대본 진행 위치

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

def vad_loop():
    global current_index
    buffer = []
    is_speaking = False
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
                    # 3. 대본과 비교
                    # ================================
                    match, score, idx = process.extractOne(
                        text, script_lines, scorer=fuzz.partial_ratio
                    )
                    if score > 50:  # 임계치 (실험적으로 조정 가능)
                        current_index = idx
                        print(f"\n📝 인식: {text}")
                        print(f"➡️ 대본[{idx}]: {match} (유사도 {score}%)")
                    else:
                        print(f"\n📝 인식: {text}")
                        print("❓ 대본과 매칭 실패")

                buffer = []
                is_speaking = False

# ================================
# 4. 실행
# ================================
import signal
def signal_handler(sig, frame):
    global running
    print("\n🛑 종료 중...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

t1 = threading.Thread(target=audio_capture, daemon=True)
t2 = threading.Thread(target=vad_loop, daemon=True)

t1.start()
t2.start()
t1.join()
t2.join()
print("✅ 종료 완료")

