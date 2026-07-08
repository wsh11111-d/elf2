import asyncio
import json
import os
import time

from aiohttp import ClientSession, WSMsgType, web


HOST = os.environ.get("HARMONY_GATEWAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("HARMONY_GATEWAY_PORT", "8080"))
API_BASE_URL = os.environ.get("HARMONY_INTERNAL_API_URL", "http://127.0.0.1:18080")
STREAM_PATH = os.environ.get("HARMONY_STREAM_PATH", "/live/stream")

clients = set()
publishers = set()
latest_frame = None
latest_frame_time = 0.0


async def stream_test_handler(request):
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>实时视频流测试</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101820;
      color: #eef5f8;
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
    }
    .status {
      padding: 6px 10px;
      border-radius: 6px;
      background: #263642;
      color: #c9d7df;
      font-size: 14px;
    }
    .screen {
      background: #05080b;
      border: 1px solid #314350;
      border-radius: 8px;
      min-height: 420px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    img {
      width: 100%;
      max-height: 74vh;
      object-fit: contain;
      display: block;
    }
    pre {
      white-space: pre-wrap;
      color: #b8c7cf;
      background: #17232c;
      border-radius: 8px;
      padding: 14px;
      margin-top: 16px;
      min-height: 56px;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>实时视频流测试</h1>
      <div id="status" class="status">connecting</div>
    </header>
    <div class="screen">
      <img id="frame" alt="等待实时视频帧">
    </div>
    <pre id="log"></pre>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const frameEl = document.getElementById("frame");
    const logEl = document.getElementById("log");
    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${location.host}/live/stream`;
    let frameCount = 0;

    function log(message) {
      logEl.textContent = `${new Date().toLocaleTimeString()} ${message}\\n` + logEl.textContent.slice(0, 1600);
    }

    function connect() {
      statusEl.textContent = `connecting ${wsUrl}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        statusEl.textContent = "connected";
        log(`connected: ${wsUrl}`);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "frame" && data.image) {
          frameCount += 1;
          frameEl.src = `data:image/jpeg;base64,${data.image}`;
          statusEl.textContent = `receiving frame ${data.frameId || frameCount}`;
          if (frameCount % 30 === 1) {
            log(`frame=${data.frameId || frameCount}, detections=${(data.detections || []).length}`);
          }
        } else {
          log(JSON.stringify(data));
        }
      };

      ws.onerror = () => {
        statusEl.textContent = "error";
        log("websocket error");
      };

      ws.onclose = () => {
        statusEl.textContent = "closed, reconnecting";
        log("closed; reconnecting in 2s");
        setTimeout(connect, 2000);
      };
    }

    connect();
  </script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def analysis_test_handler(request):
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>图片分析列表测试</title>
  <style>
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f7fa;
      color: #17212b;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 24px; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 9px 14px;
      color: white;
      background: #176b87;
      cursor: pointer;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    button.danger { background: #b42318; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 16px;
    }
    article {
      background: white;
      border: 1px solid #d8e0e7;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
    }
    img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: #dce5eb;
    }
    .image-link {
      display: block;
      min-height: 160px;
      background: #dce5eb;
    }
    .image-error {
      display: none;
      padding: 18px;
      color: #8a3b2f;
      line-height: 1.6;
      word-break: break-all;
    }
    .body { padding: 12px; }
    .title { font-weight: 700; margin-bottom: 6px; }
    .meta { color: #5b6b78; font-size: 13px; line-height: 1.6; }
    pre {
      white-space: pre-wrap;
      background: #17212b;
      color: #d8edf5;
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 16px;
      min-height: 44px;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>图片分析列表测试</h1>
      <div class="actions">
        <button id="reload">刷新</button>
        <button id="clear-history" class="danger">清空历史记录</button>
      </div>
    </header>
    <pre id="log">loading</pre>
    <section id="list" class="grid"></section>
  </main>
  <script>
    const logEl = document.getElementById("log");
    const listEl = document.getElementById("list");
    const reloadBtn = document.getElementById("reload");
    const clearBtn = document.getElementById("clear-history");
    let adminToken = "";

    function log(message) {
      logEl.textContent = message;
    }

    async function loadList() {
      listEl.innerHTML = "";
      log("登录中...");
      const loginResp = await fetch("/api/auth/login", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username: "admin001", password: "123456", role: "admin"})
      });
      const loginData = await loginResp.json();
      if (loginData.code !== 200) {
        log(JSON.stringify(loginData, null, 2));
        return;
      }
      adminToken = loginData.data.token;

      log("拉取图片分析列表...");
      const listResp = await fetch("/api/analysis/images?page=1&pageSize=20", {
        headers: {Authorization: `Bearer ${adminToken}`}
      });
      const listData = await listResp.json();
      if (listData.code !== 200) {
        log(JSON.stringify(listData, null, 2));
        return;
      }

      const items = listData.data.list || [];
      log(`total=${listData.data.total}, loaded=${items.length}`);
      for (const item of items) {
        const card = document.createElement("article");
        card.innerHTML = `
          <a class="image-link" href="${item.resultImageUrl || item.imageUrl}" target="_blank">
            <img src="${item.resultImageUrl || item.imageUrl}" alt="${item.className}">
            <div class="image-error">图片加载失败，点击打开原链接：${item.resultImageUrl || item.imageUrl}</div>
          </a>
          <div class="body">
            <div class="title">${item.className}</div>
            <div class="meta">违规行为：${(item.violationDetails || []).map(v => `${v.className} ${Math.round((v.confidence || 0) * 100)}%`).join("，") || item.className}</div>
            <div class="meta">时间：${item.createTime}</div>
            <div class="meta">人员：${item.username}</div>
          </div>
        `;
        const img = card.querySelector("img");
        const error = card.querySelector(".image-error");
        img.onerror = () => {
          img.style.display = "none";
          error.style.display = "block";
        };
        listEl.appendChild(card);
      }
    }

    async function clearHistory() {
      if (!confirm("确认清空所有图片分析历史记录和违规截图？此操作不可恢复。")) {
        return;
      }
      if (!adminToken) {
        await loadList();
      }
      log("正在清空历史记录...");
      const resp = await fetch("/api/analysis/images/clear", {
        method: "POST",
        headers: {Authorization: `Bearer ${adminToken}`}
      });
      const data = await resp.json();
      if (data.code !== 200) {
        log(JSON.stringify(data, null, 2));
        return;
      }
      log(`已清空历史记录，删除文件 ${data.data.deletedFiles} 个。`);
      await loadList();
    }

    reloadBtn.addEventListener("click", loadList);
    clearBtn.addEventListener("click", () => clearHistory().catch((err) => log(err.stack || String(err))));
    loadList().catch((err) => log(err.stack || String(err)));
  </script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def stream_handler(request):
    global latest_frame, latest_frame_time

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    clients.add(ws)
    role = "client"

    await ws.send_json({
        "type": "stream-status",
        "status": "connected",
        "clients": len(clients),
        "publishers": len(publishers),
        "timestamp": time.time(),
    })
    if latest_frame is not None:
        await ws.send_str(latest_frame)

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            if payload.get("type") == "frame":
                if role != "publisher":
                    role = "publisher"
                    publishers.add(ws)
                latest_frame = msg.data
                latest_frame_time = time.time()
                targets = [client for client in clients if client is not ws and not client.closed]
                if targets:
                    await asyncio.gather(
                        *(target.send_str(msg.data) for target in targets),
                        return_exceptions=True,
                    )
            elif payload.get("type") == "ping":
                await ws.send_json({
                    "type": "pong",
                    "timestamp": time.time(),
                    "latestFrameTime": latest_frame_time,
                })
    finally:
        clients.discard(ws)
        publishers.discard(ws)
    return ws


async def proxy_handler(request):
    target_url = f"{API_BASE_URL}{request.rel_url}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    body = await request.read()

    async with request.app["client"].request(
        request.method,
        target_url,
        headers=headers,
        data=body,
        allow_redirects=False,
    ) as resp:
        response_headers = {
            key: value
            for key, value in resp.headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
        }
        data = await resp.read()
        return web.Response(status=resp.status, body=data, headers=response_headers)


async def on_startup(app):
    app["client"] = ClientSession()


async def on_cleanup(app):
    await app["client"].close()


def create_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/stream-test", stream_test_handler)
    app.router.add_get("/analysis-test", analysis_test_handler)
    app.router.add_get(STREAM_PATH, stream_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    return app


if __name__ == "__main__":
    print(f"[INFO] Harmony gateway listening on http://{HOST}:{PORT}")
    print(f"[INFO] Proxy HTTP API to {API_BASE_URL}")
    print(f"[INFO] WebSocket stream path: ws://{HOST}:{PORT}{STREAM_PATH}")
    web.run_app(create_app(), host=HOST, port=PORT)
