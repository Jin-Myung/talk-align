import asyncio
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
clients: Set[WebSocket] = set()

# static file serving
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
def root():
    return HTMLResponse("""
<!doctype html>
<head>
    <meta charset="utf-8">
    <link rel="icon" type="image/png" href="/public/favicon.png">
</head>
<h3>Talk Align</h3>
<ul>
  <li><a href="/public/operator.html">Operator View</a></li>
  <li><a href="/public/audience.html">Audience View</a></li>
</ul>
""")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            await broadcast(msg)  # simple broker: broadcast received messages
    except WebSocketDisconnect:
        clients.discard(ws)
    except Exception:
        clients.discard(ws)

async def broadcast(message: str):
    dead = []
    for w in list(clients):
        try:
            await w.send_text(message)
        except Exception:
            dead.append(w)
    for w in dead:
        clients.discard(w)
