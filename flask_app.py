"""
Flask视频流应用 - 实验室安全检测
实时检测违规行为，推送视频流到远程WebSocket服务器。
"""
import json
import os
import time
import threading
import traceback
import base64
import asyncio
from collections import deque
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import URLError

import cv2
from flask import Flask, Response, abort, jsonify, render_template_string, send_from_directory

from func.func_yolov8_optimize import CLASSES, infer_image
from rknnpool.rknnpool_ld import initRKNN

app = Flask(__name__)

# 配置
DEVICE = "/dev/video21"
MODEL_PATH = str(Path(__file__).parent / "rknnModel" / "best_3.rknn")
SAVE_DIR = Path(__file__).parent / "results" / "violations"
EVENT_LOG_PATH = Path(__file__).parent / "results" / "violation_events.jsonl"
VIDEO_LIBRARY_DIR = Path(__file__).parent / "videos"
SAVE_INTERVAL = 10
CONFIDENCE_THRESHOLD = 0.6
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
WS_STREAM_FPS = float(os.environ.get("HARMONY_STREAM_FPS", "2"))
WS_STREAM_WIDTH = int(os.environ.get("HARMONY_STREAM_WIDTH", "320"))
WS_JPEG_QUALITY = int(os.environ.get("HARMONY_STREAM_JPEG_QUALITY", "35"))
WS_LOG_INTERVAL = float(os.environ.get("HARMONY_STREAM_LOG_INTERVAL", "10"))
SNAPSHOT_JPEG_QUALITY = int(os.environ.get("HARMONY_SNAPSHOT_JPEG_QUALITY", "95"))
ANALYSIS_SYNC_URL = os.environ.get(
    "HARMONY_ANALYSIS_SYNC_URL",
    "http://192.168.3.209:8080/api/analysis/events/sync",
)
ANALYSIS_SYNC_INTERVAL = float(os.environ.get("HARMONY_ANALYSIS_SYNC_INTERVAL", "10"))
ANALYSIS_SYNC_STATE_PATH = Path(
    os.environ.get("HARMONY_ANALYSIS_SYNC_STATE_PATH", str(Path(__file__).parent / "results" / ".analysis_sync_offset"))
)

# 远程服务器配置
WS_SERVER_URL = os.environ.get("HARMONY_STREAM_WS_URL", "ws://192.168.3.209:8080/live/stream")
WS_RECONNECT_INTERVAL = 5  # 重连间隔(秒)

# 违规判定参数
VIOLATION_WINDOW_SEC = 10.0
VIOLATION_THRESHOLD = 0.5
VIOLATION_COOLDOWN = 5.0
EVENT_COOLDOWN_SEC = 60.0

# 违规类别索引：饮水、进食、未戴防护、明火等需要进入图片分析列表。
VIOLATION_INDICES = {0, 1, 7, 8, 9, 10, 11, 30}
VIOLATION_NAMES = {
    0: "饮水",
    1: "进食",
    7: "未戴手套",
    8: "未戴头罩",
    9: "未穿实验服",
    10: "未戴口罩",
    11: "未戴护目镜",
    30: "明火",
}

# 全局变量
cap = None
rknn_lite = None
frame_count = 0
llm_result = {"text": ""}

# 违规统计
violation_stats = {vid: deque() for vid in VIOLATION_INDICES}
event_cooldown = {vid: 0.0 for vid in VIOLATION_INDICES}

# WebSocket 连接状态
ws_connected = False
ws_lock = threading.Lock()


def read_sync_offset():
    try:
        return int(ANALYSIS_SYNC_STATE_PATH.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0


def write_sync_offset(offset):
    ANALYSIS_SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_SYNC_STATE_PATH.write_text(str(offset), encoding="utf-8")


def read_new_events(offset):
    if not EVENT_LOG_PATH.exists():
        return [], offset

    size = EVENT_LOG_PATH.stat().st_size
    if offset > size:
        offset = 0

    events = []
    with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                print(f"[SYNC] 跳过无效事件行: {raw[:120]}")
        new_offset = handle.tell()
    return events, new_offset


def post_analysis_events(events):
    body = json.dumps({"events": events}, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(
        ANALYSIS_SYNC_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(req, timeout=8) as resp:
        return resp.read().decode("utf-8", errors="replace")


def analysis_sync_loop():
    offset = read_sync_offset()
    while True:
        try:
            events, new_offset = read_new_events(offset)
            if events:
                response = post_analysis_events(events)
                write_sync_offset(new_offset)
                offset = new_offset
                print(f"[SYNC] 已上传图片分析事件 {len(events)} 条: {response}")
            else:
                write_sync_offset(new_offset)
                offset = new_offset
        except URLError as exc:
            print(f"[SYNC] 上传失败: {exc}")
        except Exception as exc:
            print(f"[SYNC] 同步异常: {exc}")
            traceback.print_exc()

        time.sleep(ANALYSIS_SYNC_INTERVAL)


def build_template_alert(violations):
    violation_names = [VIOLATION_NAMES.get(v, str(v)) for v in violations]
    return f"检测到违规：{', '.join(violation_names)}，请立即整改。"


def build_status_data():
    return {
        "classes": list(CLASSES),
        "device": DEVICE,
        "save_interval": SAVE_INTERVAL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "violation_window_sec": VIOLATION_WINDOW_SEC,
        "violation_threshold": VIOLATION_THRESHOLD,
        "event_log_path": str(EVENT_LOG_PATH),
        "ws_server": WS_SERVER_URL,
        "ws_connected": ws_connected,
        "video_library_dir": str(VIDEO_LIBRARY_DIR),
    }


def list_local_videos():
    VIDEO_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for path in sorted(VIDEO_LIBRARY_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            continue

        stats = path.stat()
        videos.append(
            {
                "name": path.name,
                "size_mb": f"{stats.st_size / (1024 * 1024):.1f} MB",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.st_mtime)),
            }
        )
    return videos


def append_event_log(event):
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def init_camera():
    global cap
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap.isOpened()


def init_model():
    global rknn_lite
    rknn_lite = initRKNN(MODEL_PATH)
    return rknn_lite is not None


def calculate_violation_ratio(history, current_time):
    if not history:
        return 0.0

    cutoff = current_time - VIOLATION_WINDOW_SEC
    samples = list(history)
    if samples[0][0] > cutoff:
        samples.insert(0, (cutoff, samples[0][1]))

    total_duration = 0.0
    violation_duration = 0.0

    for index, (sample_time, is_violation) in enumerate(samples):
        next_time = samples[index + 1][0] if index + 1 < len(samples) else current_time
        start_time = max(sample_time, cutoff)
        end_time = min(next_time, current_time)
        duration = end_time - start_time
        if duration <= 0:
            continue
        total_duration += duration
        if is_violation:
            violation_duration += duration

    if total_duration <= 0:
        return 1.0 if samples[-1][1] else 0.0
    return violation_duration / total_duration


def update_violation_stats(detected_violations, current_time):
    detected_set = set(detected_violations)
    for vid in VIOLATION_INDICES:
        history = violation_stats[vid]
        history.append((current_time, vid in detected_set))

        cutoff = current_time - VIOLATION_WINDOW_SEC
        while len(history) > 1 and history[1][0] < cutoff:
            history.popleft()

    confirmed = []
    for vid in VIOLATION_INDICES:
        ratio = calculate_violation_ratio(violation_stats[vid], current_time)
        if ratio >= VIOLATION_THRESHOLD:
            confirmed.append(vid)
    return confirmed


def save_violation_frame(original_frame, result_frame, violations, alert_text):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_path = SAVE_DIR / f"{timestamp}_original.jpg"
    result_path = SAVE_DIR / f"{timestamp}_violation.jpg"

    quality = max(80, min(100, SNAPSHOT_JPEG_QUALITY))
    cv2.imwrite(str(original_path), original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    alert_frame = result_frame.copy()

    cv2.imwrite(str(result_path), alert_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    print(f"[ALERT] 原图已保存: {original_path}")
    print(f"[ALERT] 识别结果图已保存: {result_path}")
    print(f"[ALERT] 播报: {alert_text}")
    return original_path, result_path


def record_violation_event(violations, current_time, original_image_path, result_image_path, violation_details=None):
    alert_text = build_template_alert(violations)
    event = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)),
        "unix_time": current_time,
        "violations": violations,
        "violation_names": [VIOLATION_NAMES.get(v, str(v)) for v in violations],
        "alert_text": alert_text,
        "image_path": str(result_image_path),
        "original_image_path": str(original_image_path),
        "result_image_path": str(result_image_path),
        "violation_details": violation_details or [],
    }
    append_event_log(event)
    return alert_text


def handle_violation_event(frame, result_image, detections, current_time, last_check_time, last_violation_time):
    high_conf = [d for d in (detections or []) if d.get("score", 0.0) >= CONFIDENCE_THRESHOLD]
    detected_violations = [d["class_id"] for d in high_conf if d.get("class_id") in VIOLATION_INDICES]
    confirmed_violations = update_violation_stats(detected_violations, current_time)

    if current_time - last_check_time < SAVE_INTERVAL:
        return last_check_time, last_violation_time

    last_check_time = current_time
    if not confirmed_violations or current_time - last_violation_time < VIOLATION_COOLDOWN:
        return last_check_time, last_violation_time

    cooldown_hit = []
    for vid in confirmed_violations:
        if current_time - event_cooldown.get(vid, 0.0) >= EVENT_COOLDOWN_SEC:
            cooldown_hit.append(vid)

    if cooldown_hit:
        current_violation_ids = {d["class_id"] for d in high_conf if d.get("class_id") in cooldown_hit}
        if not current_violation_ids:
            return last_check_time, last_violation_time

        event_violations = sorted(current_violation_ids)
        violation_details = []
        for vid in event_violations:
            best = max(
                (d for d in high_conf if d.get("class_id") == vid),
                key=lambda item: item.get("score", 0.0),
                default=None,
            )
            violation_details.append({
                "classId": vid,
                "className": VIOLATION_NAMES.get(vid, str(vid)),
                "confidence": float(best.get("score", 0.0)) if best else 0.0,
            })

        for vid in event_violations:
            event_cooldown[vid] = current_time

        alert_text = build_template_alert(event_violations)
        original_path, result_path = save_violation_frame(
            frame,
            result_image,
            event_violations,
            alert_text,
        )
        llm_result["text"] = alert_text
        record_violation_event(event_violations, current_time, original_path, result_path, violation_details)
        print(f"[EVENT] 已记录违规事件: {[VIOLATION_NAMES.get(v, v) for v in event_violations]}")

    return last_check_time, current_time


async def send_to_websocket(message: dict, ws_client):
    """发送数据到WebSocket服务器"""
    await ws_client.send_str(json.dumps(message))


def normalize_detection(d):
    class_id = int(d.get("class_id", -1))
    box = d.get("box")
    if box and len(box) == 4:
        top, left, right, bottom = [int(v) for v in box]
        x = left
        y = top
        width = max(0, right - left)
        height = max(0, bottom - top)
    else:
        x = int(d.get("x", 0))
        y = int(d.get("y", 0))
        width = int(d.get("w", d.get("width", 0)))
        height = int(d.get("h", d.get("height", 0)))

    return {
        "classId": class_id,
        "className": CLASSES[class_id] if 0 <= class_id < len(CLASSES) else str(class_id),
        "confidence": float(d.get("score", d.get("confidence", 0.0))),
        "bbox": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
    }


def prepare_stream_frame(frame):
    if WS_STREAM_WIDTH > 0 and frame.shape[1] > WS_STREAM_WIDTH:
        scale = WS_STREAM_WIDTH / frame.shape[1]
        target_height = max(1, int(frame.shape[0] * scale))
        frame = cv2.resize(frame, (WS_STREAM_WIDTH, target_height), interpolation=cv2.INTER_AREA)

    quality = max(10, min(95, WS_JPEG_QUALITY))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode stream frame")
    return base64.b64encode(buffer).decode("utf-8"), len(buffer)


def build_frame_message(frame, detections, frame_id, timestamp):
    """构建推送消息"""
    detections = detections or []
    frame_base64, jpeg_bytes = prepare_stream_frame(frame)

    # 统计各违规类别数量
    class_stats = {}
    for d in detections:
        name = CLASSES[d["class_id"]] if d["class_id"] < len(CLASSES) else str(d["class_id"])
        class_stats[name] = class_stats.get(name, 0) + 1

    return {
        "type": "frame",
        "timestamp": timestamp,
        "frameId": frame_id,
        "jpegBytes": jpeg_bytes,
        "image": frame_base64,  # Base64编码的JPEG
        "detections": [normalize_detection(d) for d in detections],
        "statistics": {
            "total": len(detections),
            "classes": class_stats,
        }
    }


async def websocket_sender_loop():
    """WebSocket发送循环"""
    global ws_connected, frame_count, llm_result

    import aiohttp

    last_send_time = 0.0
    send_interval = 1.0 / max(0.1, WS_STREAM_FPS)
    last_log_time = 0.0
    bytes_since_log = 0
    last_check_time = 0.0
    last_violation_time = 0.0

    while True:
        try:
            print(f"[WS] 正在连接 {WS_SERVER_URL}...")
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(WS_SERVER_URL, heartbeat=30) as ws_client:
                    ws_connected = True
                    print("[WS] 连接成功")

                    while True:
                        # 等待摄像头和模型就绪
                        if cap is None or rknn_lite is None:
                            await asyncio.sleep(1)
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            await asyncio.sleep(0.1)
                            continue

                        frame_count += 1
                        current_time = time.time()
                        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_time))

                        try:
                            result_image, detections, _, _ = infer_image(
                                rknn_lite,
                                frame,
                                draw_result=True,
                                return_detections=True,
                                post_process=True,
                                return_profile=False,
                            )
                        except Exception as infer_exc:
                            print(f"[WS] 单帧推理异常，发送原始帧: {infer_exc}")
                            result_image, detections = frame, []

                        last_check_time, last_violation_time = handle_violation_event(
                            frame,
                            result_image,
                            detections,
                            current_time,
                            last_check_time,
                            last_violation_time,
                        )

                        # 速率控制
                        if current_time - last_send_time < send_interval:
                            await asyncio.sleep(0.01)
                            continue

                        # 构建并发送消息
                        message = build_frame_message(result_image, detections, frame_count, timestamp)
                        await send_to_websocket(message, ws_client)
                        last_send_time = current_time
                        bytes_since_log += message["jpegBytes"]
                        if current_time - last_log_time >= WS_LOG_INTERVAL:
                            elapsed = max(0.001, current_time - last_log_time) if last_log_time else WS_LOG_INTERVAL
                            kbps = (bytes_since_log * 8) / elapsed / 1000
                            print(
                                f"[WS] 推流: fps={WS_STREAM_FPS}, width={WS_STREAM_WIDTH}, "
                                f"quality={WS_JPEG_QUALITY}, last_jpeg={message['jpegBytes']}B, "
                                f"avg={kbps:.1f}kbps"
                            )
                            last_log_time = current_time
                            bytes_since_log = 0

        except Exception as e:
            print(f"[WS] 连接或发送异常: {e}")
            traceback.print_exc()
        finally:
            ws_connected = False

        await asyncio.sleep(WS_RECONNECT_INTERVAL)


def websocket_sender():
    """WebSocket发送线程"""
    asyncio.run(websocket_sender_loop())


def generate_frames():
    """本地视频流生成（用于调试）"""
    global frame_count

    last_check_time = 0.0
    last_violation_time = 0.0

    print("[INFO] 开始生成视频流...")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 无法读取帧，尝试重新打开摄像头...")
                cap.release()
                time.sleep(1)
                cap.open(DEVICE, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                continue

            frame_count += 1
            current_time = time.time()

            result_image, detections, _, _ = infer_image(
                rknn_lite,
                frame,
                draw_result=True,
                return_detections=True,
                post_process=True,
                return_profile=False,
            )

            last_check_time, last_violation_time = handle_violation_event(
                frame,
                result_image,
                detections,
                current_time,
                last_check_time,
                last_violation_time,
            )

            ret, buffer = cv2.imencode(".jpg", result_image)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            time.sleep(0.033)

        except Exception as e:
            print(f"[ERROR] 视频流异常: {e}")
            traceback.print_exc()
            time.sleep(1)


@app.route("/")
def index():
    return render_template_string(HOME_TEMPLATE, status=build_status_data())


@app.route("/live")
def live():
    return render_template_string(LIVE_TEMPLATE, status=build_status_data())


@app.route("/videos")
def videos():
    return render_template_string(
        VIDEO_LIBRARY_TEMPLATE,
        status=build_status_data(),
        videos=list_local_videos(),
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/videos/files/<path:filename>")
def serve_video(filename):
    video_path = VIDEO_LIBRARY_DIR / filename
    if video_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        abort(404)
    if not video_path.exists() or not video_path.is_file():
        abort(404)
    return send_from_directory(str(VIDEO_LIBRARY_DIR), filename)


@app.route("/status")
def status():
    current_time = time.time()
    stats = {}
    for vid in VIOLATION_INDICES:
        name = VIOLATION_NAMES.get(vid, str(vid))
        ratio = calculate_violation_ratio(violation_stats[vid], current_time)
        stats[name] = f"{ratio:.1%}"

    return jsonify(
        {
            "classes": list(CLASSES),
            "device": DEVICE,
            "save_interval": SAVE_INTERVAL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "violation_stats": stats,
            "frame_count": frame_count,
            "latest_alert": llm_result.get("text", ""),
            "event_log_path": str(EVENT_LOG_PATH),
            "ws_server": WS_SERVER_URL,
            "ws_connected": ws_connected,
        }
    )


HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>化工实验室操作系统</title>
    <style>
        :root {
            --bg: #08131a;
            --bg-soft: rgba(8, 19, 26, 0.78);
            --panel: rgba(14, 35, 46, 0.88);
            --panel-strong: rgba(18, 47, 61, 0.96);
            --line: rgba(120, 192, 214, 0.18);
            --text: #e7f8ff;
            --muted: #8fb0bd;
            --accent: #5ee6d0;
            --accent-2: #7bb4ff;
            --warning: #ffb870;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top right, rgba(94, 230, 208, 0.18), transparent 26%),
                radial-gradient(circle at left center, rgba(123, 180, 255, 0.18), transparent 22%),
                linear-gradient(135deg, #051018 0%, #0a1d27 52%, #061119 100%);
        }
        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(123, 180, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(123, 180, 255, 0.05) 1px, transparent 1px);
            background-size: 44px 44px;
            pointer-events: none;
            mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.5), transparent 90%);
        }
        .shell {
            position: relative;
            z-index: 1;
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 40px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(7, 21, 30, 0.74);
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .badge {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, rgba(94, 230, 208, 0.28), rgba(123, 180, 255, 0.22));
            border: 1px solid rgba(94, 230, 208, 0.32);
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .brand h1,
        .hero h2,
        .section-title,
        .option-card h3,
        .metric strong {
            margin: 0;
        }
        .brand p,
        .hero p,
        .option-card p,
        .metric span,
        .status-chip,
        .panel-note {
            margin: 0;
            color: var(--muted);
        }
        .status-chip {
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid rgba(94, 230, 208, 0.24);
            background: rgba(94, 230, 208, 0.09);
            font-size: 14px;
        }
        .hero {
            margin-top: 24px;
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
            gap: 22px;
        }
        .hero-card,
        .side-panel,
        .option-card {
            border: 1px solid var(--line);
            border-radius: 30px;
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow);
        }
        .hero-card {
            padding: 34px;
            position: relative;
            overflow: hidden;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            right: -60px;
            top: -60px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(94, 230, 208, 0.2), transparent 72%);
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid rgba(123, 180, 255, 0.2);
            background: rgba(123, 180, 255, 0.08);
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero h2 {
            margin-top: 18px;
            font-size: clamp(34px, 6vw, 58px);
            line-height: 1.04;
            letter-spacing: 0.04em;
        }
        .hero p {
            margin-top: 14px;
            max-width: 620px;
            font-size: 16px;
            line-height: 1.7;
        }
        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
            margin-top: 28px;
        }
        .primary-btn,
        .ghost-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 54px;
            padding: 0 24px;
            border-radius: 16px;
            text-decoration: none;
            font-weight: 700;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .primary-btn {
            color: #052229;
            background: linear-gradient(135deg, var(--accent), #8ef8e0);
            box-shadow: 0 12px 30px rgba(94, 230, 208, 0.2);
        }
        .ghost-btn {
            color: var(--text);
            border: 1px solid rgba(123, 180, 255, 0.22);
            background: rgba(123, 180, 255, 0.08);
        }
        .primary-btn:hover,
        .ghost-btn:hover {
            transform: translateY(-2px);
        }
        .side-panel {
            padding: 26px;
            background:
                linear-gradient(180deg, rgba(18, 47, 61, 0.96), rgba(8, 20, 29, 0.95)),
                var(--panel-strong);
        }
        .section-title {
            font-size: 18px;
            letter-spacing: 0.06em;
        }
        .metrics {
            margin-top: 20px;
            display: grid;
            gap: 14px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(255, 255, 255, 0.03);
        }
        .metric strong {
            font-size: 15px;
        }
        .metric span:last-child {
            color: var(--text);
            text-align: right;
        }
        .options {
            margin-top: 26px;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 22px;
        }
        .option-card {
            padding: 26px;
            text-decoration: none;
            color: inherit;
            position: relative;
            overflow: hidden;
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
        }
        .option-card::before {
            content: "";
            position: absolute;
            inset: auto -40px -60px auto;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            opacity: 0.7;
        }
        .option-card.live::before {
            background: radial-gradient(circle, rgba(94, 230, 208, 0.22), transparent 72%);
        }
        .option-card.library::before {
            background: radial-gradient(circle, rgba(255, 184, 112, 0.18), transparent 72%);
        }
        .option-card:hover {
            transform: translateY(-4px);
            border-color: rgba(123, 180, 255, 0.34);
            box-shadow: 0 26px 60px rgba(0, 0, 0, 0.28);
        }
        .option-tag {
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            margin-bottom: 18px;
        }
        .live .option-tag {
            color: #9ef3e3;
            background: rgba(94, 230, 208, 0.12);
        }
        .library .option-tag {
            color: #ffd4aa;
            background: rgba(255, 184, 112, 0.12);
        }
        .option-card h3 {
            font-size: 28px;
            letter-spacing: 0.04em;
        }
        .option-card p {
            margin-top: 12px;
            line-height: 1.7;
        }
        .option-arrow {
            margin-top: 24px;
            color: var(--text);
            font-weight: 700;
        }
        .panel-note {
            margin-top: 18px;
            font-size: 14px;
            line-height: 1.7;
        }
        @media (max-width: 980px) {
            .hero,
            .options {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 640px) {
            .shell {
                width: min(100% - 20px, 1180px);
                padding-top: 16px;
            }
            .topbar {
                flex-direction: column;
                align-items: flex-start;
            }
            .hero-card,
            .side-panel,
            .option-card {
                border-radius: 22px;
                padding: 20px;
            }
            .hero h2 {
                font-size: 32px;
            }
            .hero-actions {
                flex-direction: column;
            }
            .primary-btn,
            .ghost-btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="topbar">
            <div class="brand">
                <div class="badge">LAB</div>
                <div>
                    <h1>化工实验室操作系统</h1>
                    <p>实验监测、实时巡检、操作视频统一入口</p>
                </div>
            </div>
            <div class="status-chip" id="ws-pill">
                WebSocket {{ "在线" if status.ws_connected else "离线" }}
            </div>
        </section>

        <section class="hero">
            <article class="hero-card">
                <div class="eyebrow">Chemical Lab Control Center</div>
                <h2>化工实验室操作系统</h2>
                <p>
                    面向实验室巡检与教学展示的统一主界面。可直接进入实时视频流展示页面，
                    也可在操作视频选取页浏览后续上传的本地视频素材。
                </p>
                <div class="hero-actions">
                    <a class="primary-btn" href="{{ url_for('live') }}">进入实时视频流展示</a>
                    <a class="ghost-btn" href="{{ url_for('videos') }}">进入操作视频选取</a>
                </div>
            </article>

            <aside class="side-panel">
                <h3 class="section-title">系统概览</h3>
                <div class="metrics">
                    <div class="metric">
                        <strong>视频设备</strong>
                        <span>{{ status.device }}</span>
                    </div>
                    <div class="metric">
                        <strong>置信度阈值</strong>
                        <span>{{ status.confidence_threshold }}</span>
                    </div>
                    <div class="metric">
                        <strong>违规判定窗口</strong>
                        <span>{{ status.violation_window_sec }} 秒</span>
                    </div>
                    <div class="metric">
                        <strong>视频库目录</strong>
                        <span>{{ status.video_library_dir }}</span>
                    </div>
                </div>
                <p class="panel-note">
                    本地操作视频后续放入视频库目录后，页面会自动读取并展示，无需额外改 UI。
                </p>
            </aside>
        </section>

        <section class="options">
            <a class="option-card live" href="{{ url_for('live') }}">
                <span class="option-tag">实时监测</span>
                <h3>实时视频流展示</h3>
                <p>进入检测直播页，查看摄像头画面、连接状态、违规告警与实时运行信息。</p>
                <div class="option-arrow">打开直播界面 →</div>
            </a>

            <a class="option-card library" href="{{ url_for('videos') }}">
                <span class="option-tag">本地素材</span>
                <h3>操作视频选取</h3>
                <p>用于展示你稍后上传的本地操作视频，支持列表浏览与单页直接播放。</p>
                <div class="option-arrow">打开视频库 →</div>
            </a>
        </section>
    </main>
    <script>
        setInterval(() => {
            fetch('/status')
                .then((r) => r.json())
                .then((data) => {
                    const pill = document.getElementById('ws-pill');
                    pill.textContent = `WebSocket ${data.ws_connected ? '在线' : '离线'}`;
                })
                .catch(() => {});
        }, 5000);
    </script>
</body>
</html>
"""


LIVE_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实时视频流展示</title>
    <style>
        :root {
            --bg: #071219;
            --panel: rgba(13, 34, 44, 0.9);
            --panel-2: rgba(9, 23, 31, 0.88);
            --line: rgba(120, 192, 214, 0.18);
            --text: #e7f8ff;
            --muted: #8fb0bd;
            --accent: #5ee6d0;
            --accent-2: #7bb4ff;
            --ok: #79f0af;
            --bad: #ff8f7e;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(94, 230, 208, 0.14), transparent 26%),
                linear-gradient(135deg, #061018 0%, #0a1b24 48%, #08131a 100%);
        }
        .shell {
            width: min(1280px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 34px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(7, 21, 30, 0.74);
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .brand-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, rgba(94, 230, 208, 0.28), rgba(123, 180, 255, 0.22));
            border: 1px solid rgba(94, 230, 208, 0.32);
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .brand h1,
        .panel h3,
        .stream-title {
            margin: 0;
        }
        .brand p,
        .subtext,
        .meta-label,
        .class-chip {
            margin: 0;
            color: var(--muted);
        }
        .nav-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .nav-btn {
            min-height: 46px;
            padding: 0 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            text-decoration: none;
            color: var(--text);
            border: 1px solid rgba(123, 180, 255, 0.2);
            background: rgba(123, 180, 255, 0.08);
        }
        .grid {
            margin-top: 24px;
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.7fr);
            gap: 22px;
        }
        .stream-card,
        .panel {
            border: 1px solid var(--line);
            border-radius: 30px;
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow);
        }
        .stream-card {
            padding: 22px;
        }
        .stream-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
        }
        .signal {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 999px;
            font-size: 14px;
            border: 1px solid rgba(94, 230, 208, 0.22);
            background: rgba(94, 230, 208, 0.08);
        }
        .signal-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--ok);
            box-shadow: 0 0 18px rgba(121, 240, 175, 0.7);
        }
        .signal.offline {
            border-color: rgba(255, 143, 126, 0.22);
            background: rgba(255, 143, 126, 0.08);
        }
        .signal.offline .signal-dot {
            background: var(--bad);
            box-shadow: 0 0 18px rgba(255, 143, 126, 0.7);
        }
        .stream-frame {
            overflow: hidden;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background:
                linear-gradient(135deg, rgba(123, 180, 255, 0.08), rgba(94, 230, 208, 0.08)),
                var(--panel-2);
            aspect-ratio: 16 / 9;
        }
        .stream-frame img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .panel {
            padding: 22px;
        }
        .panel + .panel {
            margin-top: 18px;
        }
        .metrics {
            display: grid;
            gap: 12px;
            margin-top: 18px;
        }
        .metric {
            padding: 15px 16px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(255, 255, 255, 0.03);
        }
        .metric strong {
            display: block;
            margin-top: 6px;
            font-size: 15px;
            color: var(--text);
            word-break: break-all;
        }
        .alert-box {
            margin-top: 18px;
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid rgba(255, 184, 112, 0.18);
            background: rgba(255, 184, 112, 0.09);
            color: #ffe1bf;
            line-height: 1.7;
            min-height: 86px;
        }
        .classes {
            margin-top: 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .class-chip {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(123, 180, 255, 0.08);
            border: 1px solid rgba(123, 180, 255, 0.16);
            font-size: 13px;
        }
        @media (max-width: 980px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 640px) {
            .shell {
                width: min(100% - 20px, 1280px);
                padding-top: 16px;
            }
            .topbar {
                flex-direction: column;
                align-items: flex-start;
            }
            .nav-actions {
                width: 100%;
            }
            .nav-btn {
                flex: 1;
            }
            .stream-card,
            .panel {
                border-radius: 22px;
                padding: 18px;
            }
            .stream-head {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="topbar">
            <div class="brand">
                <div class="brand-mark">LAB</div>
                <div>
                    <h1>实时视频流展示</h1>
                    <p>化工实验室操作系统 / 在线检测界面</p>
                </div>
            </div>
            <div class="nav-actions">
                <a class="nav-btn" href="{{ url_for('index') }}">返回主界面</a>
                <a class="nav-btn" href="{{ url_for('videos') }}">操作视频选取</a>
            </div>
        </section>

        <section class="grid">
            <article class="stream-card">
                <div class="stream-head">
                    <div>
                        <h2 class="stream-title">实验室实时监测画面</h2>
                        <p class="subtext">摄像头检测结果已按主界面风格统一展示</p>
                    </div>
                    <div id="ws-indicator" class="signal {{ '' if status.ws_connected else 'offline' }}">
                        <span class="signal-dot"></span>
                        <span id="ws-text">{{ "WebSocket 在线" if status.ws_connected else "WebSocket 离线" }}</span>
                    </div>
                </div>
                <div class="stream-frame">
                    <img src="{{ url_for('video_feed') }}" alt="实验室实时视频流">
                </div>
            </article>

            <aside>
                <section class="panel">
                    <h3>运行状态</h3>
                    <div class="metrics">
                        <div class="metric">
                            <p class="meta-label">视频设备</p>
                            <strong>{{ status.device }}</strong>
                        </div>
                        <div class="metric">
                            <p class="meta-label">WebSocket 服务</p>
                            <strong>{{ status.ws_server }}</strong>
                        </div>
                        <div class="metric">
                            <p class="meta-label">事件日志</p>
                            <strong>{{ status.event_log_path }}</strong>
                        </div>
                        <div class="metric">
                            <p class="meta-label">当前帧计数</p>
                            <strong id="frame-count">0</strong>
                        </div>
                    </div>
                </section>

                <section class="panel">
                    <h3>告警信息</h3>
                    <div id="latest-alert" class="alert-box">当前暂无新的违规告警。</div>
                    <div class="metrics">
                        <div class="metric">
                            <p class="meta-label">置信度阈值</p>
                            <strong>{{ status.confidence_threshold }}</strong>
                        </div>
                        <div class="metric">
                            <p class="meta-label">违规判定窗口</p>
                            <strong>{{ status.violation_window_sec }} 秒 / {{ (status.violation_threshold * 100)|int }}%</strong>
                        </div>
                    </div>
                </section>
            </aside>
        </section>

        <section class="panel">
            <h3>检测类别</h3>
            <div class="classes">
                {% for i in range(status.classes|length) %}
                <span class="class-chip">{{ i }} · {{ status.classes[i] }}</span>
                {% endfor %}
            </div>
        </section>
    </main>
    <script>
        const wsIndicator = document.getElementById('ws-indicator');
        const wsText = document.getElementById('ws-text');
        const frameCount = document.getElementById('frame-count');
        const latestAlert = document.getElementById('latest-alert');

        function updateStatus() {
            fetch('/status')
                .then((r) => r.json())
                .then((data) => {
                    frameCount.textContent = data.frame_count;
                    latestAlert.textContent = data.latest_alert || '当前暂无新的违规告警。';
                    wsText.textContent = data.ws_connected ? 'WebSocket 在线' : 'WebSocket 离线';
                    wsIndicator.classList.toggle('offline', !data.ws_connected);
                })
                .catch(() => {
                    wsText.textContent = '状态获取失败';
                    wsIndicator.classList.add('offline');
                });
        }

        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
"""


VIDEO_LIBRARY_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>操作视频选取</title>
    <style>
        :root {
            --bg: #09131a;
            --panel: rgba(13, 34, 44, 0.9);
            --line: rgba(120, 192, 214, 0.18);
            --text: #e7f8ff;
            --muted: #8fb0bd;
            --accent: #ffb870;
            --accent-2: #7bb4ff;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Avenir Next", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top right, rgba(255, 184, 112, 0.14), transparent 24%),
                radial-gradient(circle at left center, rgba(123, 180, 255, 0.14), transparent 22%),
                linear-gradient(135deg, #061018 0%, #0a1b24 48%, #08131a 100%);
        }
        .shell {
            width: min(1240px, calc(100% - 32px));
            margin: 0 auto;
            padding: 28px 0 34px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding: 16px 20px;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(7, 21, 30, 0.74);
            backdrop-filter: blur(12px);
            box-shadow: var(--shadow);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .brand-mark {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, rgba(255, 184, 112, 0.22), rgba(123, 180, 255, 0.22));
            border: 1px solid rgba(255, 184, 112, 0.28);
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .brand h1,
        .intro h2,
        .video-card h3,
        .empty-card h3 {
            margin: 0;
        }
        .brand p,
        .intro p,
        .video-meta,
        .empty-card p,
        .path-box {
            margin: 0;
            color: var(--muted);
        }
        .nav-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .nav-btn {
            min-height: 46px;
            padding: 0 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            text-decoration: none;
            color: var(--text);
            border: 1px solid rgba(123, 180, 255, 0.2);
            background: rgba(123, 180, 255, 0.08);
        }
        .intro,
        .path-box,
        .video-card,
        .empty-card {
            border: 1px solid var(--line);
            border-radius: 28px;
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow);
        }
        .intro {
            margin-top: 24px;
            padding: 28px;
        }
        .intro h2 {
            font-size: clamp(28px, 5vw, 42px);
            letter-spacing: 0.04em;
        }
        .intro p {
            margin-top: 12px;
            line-height: 1.8;
            max-width: 860px;
        }
        .path-box {
            margin-top: 18px;
            padding: 18px 20px;
            font-size: 14px;
            line-height: 1.8;
        }
        .path-box strong {
            color: var(--text);
        }
        .library-grid {
            margin-top: 22px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }
        .video-card,
        .empty-card {
            padding: 18px;
        }
        .video-card video {
            width: 100%;
            aspect-ratio: 16 / 9;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: #061018;
            object-fit: cover;
        }
        .video-card h3 {
            margin-top: 16px;
            font-size: 20px;
            line-height: 1.5;
            word-break: break-all;
        }
        .video-meta {
            margin-top: 8px;
            font-size: 14px;
            line-height: 1.7;
        }
        .video-link {
            display: inline-flex;
            margin-top: 16px;
            padding: 10px 14px;
            border-radius: 14px;
            color: #ffe6ca;
            text-decoration: none;
            background: rgba(255, 184, 112, 0.1);
            border: 1px solid rgba(255, 184, 112, 0.18);
        }
        .empty-card {
            margin-top: 22px;
            padding: 28px;
            text-align: center;
        }
        .empty-card p {
            margin-top: 12px;
            line-height: 1.8;
        }
        @media (max-width: 640px) {
            .shell {
                width: min(100% - 20px, 1240px);
                padding-top: 16px;
            }
            .topbar {
                flex-direction: column;
                align-items: flex-start;
            }
            .nav-actions {
                width: 100%;
            }
            .nav-btn {
                flex: 1;
            }
            .intro,
            .path-box,
            .video-card,
            .empty-card {
                border-radius: 22px;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="topbar">
            <div class="brand">
                <div class="brand-mark">VID</div>
                <div>
                    <h1>操作视频选取</h1>
                    <p>化工实验室操作系统 / 本地视频库</p>
                </div>
            </div>
            <div class="nav-actions">
                <a class="nav-btn" href="{{ url_for('index') }}">返回主界面</a>
                <a class="nav-btn" href="{{ url_for('live') }}">实时视频流展示</a>
            </div>
        </section>

        <section class="intro">
            <h2>本地操作视频展示区</h2>
            <p>
                这里用于展示你稍后上传的本地操作视频。页面会自动扫描视频目录，
                并以统一卡片样式展示，支持直接预览与单独打开播放。
            </p>
        </section>

        <section class="path-box">
            <strong>视频放置目录：</strong>{{ status.video_library_dir }}<br>
            支持格式：mp4、mov、avi、mkv、webm、m4v
        </section>

        {% if videos %}
        <section class="library-grid">
            {% for video in videos %}
            <article class="video-card">
                <video controls preload="metadata">
                    <source src="{{ url_for('serve_video', filename=video.name) }}">
                    当前浏览器不支持视频播放。
                </video>
                <h3>{{ video.name }}</h3>
                <p class="video-meta">大小：{{ video.size_mb }}</p>
                <p class="video-meta">更新时间：{{ video.updated_at }}</p>
                <a class="video-link" href="{{ url_for('serve_video', filename=video.name) }}" target="_blank">单独打开视频</a>
            </article>
            {% endfor %}
        </section>
        {% else %}
        <section class="empty-card">
            <h3>视频库当前为空</h3>
            <p>
                你把本地操作视频上传到 <strong>{{ status.video_library_dir }}</strong> 后，
                刷新本页即可自动显示，不需要再改界面。
            </p>
        </section>
        {% endif %}
    </main>
</body>
</html>
"""


def main():
    print("正在初始化...")
    print(f"正在打开摄像头: {DEVICE}")
    if not init_camera():
        raise RuntimeError(f"无法打开摄像头: {DEVICE}")

    print(f"正在加载模型: {MODEL_PATH}")
    if not init_model():
        raise RuntimeError(f"无法加载模型: {MODEL_PATH}")

    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 启动WebSocket发送线程
    ws_thread = threading.Thread(target=websocket_sender, daemon=True)
    ws_thread.start()
    print(f"[INFO] WebSocket推送线程已启动，目标: {WS_SERVER_URL}")

    sync_thread = threading.Thread(target=analysis_sync_loop, daemon=True)
    sync_thread.start()
    print(f"[INFO] 图片分析事件同步线程已启动，目标: {ANALYSIS_SYNC_URL}, 间隔: {ANALYSIS_SYNC_INTERVAL}s")

    print("启动Flask服务器...")
    print("访问 http://0.0.0.0:5000 查看本地视频流")
    print("视频流同时推送至:", WS_SERVER_URL)
    print("违规图片将保存到:", SAVE_DIR)
    print("违规事件日志:", EVENT_LOG_PATH)

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
