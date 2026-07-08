"""Deprecated standalone 8081 stream server.

The active deployment uses harmony_gateway_8080.py so HTTP APIs and
WebSocket stream share ws://192.168.3.209:8080/live/stream.
"""

import asyncio
import json
import os
import time

import websockets


HOST = os.environ.get("HARMONY_STREAM_HOST", "0.0.0.0")
PORT = int(os.environ.get("HARMONY_STREAM_PORT", "8081"))
PATH = os.environ.get("HARMONY_STREAM_PATH", "/live/stream")

clients = set()
publishers = set()
latest_frame = None
latest_frame_time = 0.0


async def send_json(ws, payload):
    await ws.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


async def handle_connection(ws):
    global latest_frame, latest_frame_time

    request_path = getattr(ws, "request", None)
    path = getattr(request_path, "path", None) or getattr(ws, "path", "")
    if path and path != PATH:
        await ws.close(code=1008, reason="invalid path")
        return

    clients.add(ws)
    role = "client"
    try:
        await send_json(ws, {
            "type": "stream-status",
            "status": "connected",
            "clients": len(clients),
            "publishers": len(publishers),
            "timestamp": time.time(),
        })
        if latest_frame is not None:
            await ws.send(latest_frame)

        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            if payload.get("type") == "frame":
                if role != "publisher":
                    role = "publisher"
                    publishers.add(ws)
                latest_frame = message
                latest_frame_time = time.time()
                targets = [client for client in clients if client is not ws]
                if targets:
                    await asyncio.gather(
                        *(target.send(message) for target in targets),
                        return_exceptions=True,
                    )
            elif payload.get("type") == "ping":
                await send_json(ws, {
                    "type": "pong",
                    "timestamp": time.time(),
                    "latestFrameTime": latest_frame_time,
                })
    finally:
        clients.discard(ws)
        publishers.discard(ws)


async def main():
    print(f"[INFO] Harmony stream server listening on ws://{HOST}:{PORT}{PATH}")
    async with websockets.serve(handle_connection, HOST, PORT, ping_interval=30, ping_timeout=10):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
