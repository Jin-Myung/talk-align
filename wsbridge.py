from websocket import create_connection, WebSocketConnectionClosedException
import json, threading, queue, time

class WSBridge:
    """Thread-safe WebSocket bridge for CLI: send results to server and receive operator commands."""
    def __init__(self, url="ws://127.0.0.1:8000/ws"):
        self.ws = create_connection(url)
        self.lock = threading.Lock()
        self.cmd_q = queue.Queue(maxsize=5)

        def receiver():
            while True:
                try:
                    data = self.ws.recv()
                    if not data:
                        time.sleep(0.05); continue
                    self.cmd_q.put(json.loads(data))
                except WebSocketConnectionClosedException:
                    break
                except Exception:
                    time.sleep(0.1)
        threading.Thread(target=receiver, daemon=True).start()

    def send(self, payload: dict):
        with self.lock:
            try:
                self.ws.send(json.dumps(payload, ensure_ascii=False))
            except WebSocketConnectionClosedException:
                pass

    def get_cmd_nowait(self):
        try:
            return self.cmd_q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        with self.lock:
            try: self.ws.close()
            except Exception: pass
