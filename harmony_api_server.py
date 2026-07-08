import base64
import hashlib
import hmac
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from flask import Flask, has_request_context, jsonify, request, send_file


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
VIOLATIONS_DIR = RESULTS_DIR / "violations"
EVENT_LOG_PATH = RESULTS_DIR / "violation_events.jsonl"
VIDEO_DIR_CANDIDATES = [ BASE_DIR / "videos"]

HOST = os.environ.get("HARMONY_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("HARMONY_API_PORT", "8080"))
PUBLIC_BASE_URL = os.environ.get("HARMONY_PUBLIC_BASE_URL", "").strip().rstrip("/")
MEDIA_BASE_URL = os.environ.get("HARMONY_MEDIA_BASE_URL", "").strip().rstrip("/")
TOKEN_SECRET = os.environ.get("HARMONY_API_SECRET", "lab-demo-secret")
TOKEN_EXPIRE_SECONDS = 7 * 24 * 60 * 60

DEFAULT_RECORD_USER_ID = os.environ.get("HARMONY_DEFAULT_USER_ID", "user-1001")
DEFAULT_RECORD_USERNAME = os.environ.get("HARMONY_DEFAULT_USERNAME", "实验员001")
ALLOWED_ANALYSIS_VIOLATIONS = {
    "未戴手套",
    "未戴头罩",
    "未穿实验服",
    "未戴口罩",
    "进食",
    "饮水",
    "明火",
    "未佩戴护目镜",
}

VIOLATION_METADATA = {
    "未佩戴护目镜": {
        "severity": "高危",
        "hazard": "化学飞溅或碎片可能直接损伤眼部，严重时可导致灼伤或失明。",
        "standard": "GB/T 32119-2015 个体防护装备 眼面部防护",
        "correct_method": [
            "进入实验区域前检查并佩戴护目镜，确保镜片完整清洁。",
            "处理腐蚀性、挥发性或易飞溅试剂时全程保持佩戴。",
            "离开实验区后再摘下，并按规定清洁消毒存放。",
        ],
    },
    "未穿实验服": {
        "severity": "中危",
        "hazard": "皮肤和日常衣物暴露在化学品环境中，增加污染和灼伤风险。",
        "standard": "GB 39800.1-2020 个体防护装备配备规范",
        "correct_method": [
            "进入实验室前穿着合规实验服并扣好衣扣。",
            "实验服污染后及时更换，不得穿出污染区域。",
            "实验结束后按规定分类回收或清洗实验服。",
        ],
    },
    "未戴手套": {
        "severity": "中危",
        "hazard": "皮肤可能直接接触有毒、有腐蚀性或刺激性化学品。",
        "standard": "GB 24541-2022 手部防护 机械危害防护手套",
        "correct_method": [
            "根据试剂类型选择匹配材质的防护手套。",
            "操作前检查手套完整性，发现破损立即更换。",
            "接触污染源后按规范脱卸手套并及时洗手。",
        ],
    },
    "未戴口罩": {
        "severity": "中危",
        "hazard": "可能吸入粉尘、气溶胶或挥发性有害气体，影响呼吸健康。",
        "standard": "GB 39800.1-2020 个体防护装备配备规范",
        "correct_method": [
            "根据实验风险佩戴医用口罩或防护口罩。",
            "确保口鼻完全覆盖并贴合面部，避免频繁触碰。",
            "口罩受潮或污染后及时更换。",
        ],
    },
    "未戴头罩": {
        "severity": "中危",
        "hazard": "头发可能接触明火、旋转部件或污染性化学品。",
        "standard": "实验室个人防护通用要求",
        "correct_method": [
            "长发人员进入实验区前应束发并佩戴头罩。",
            "在明火、搅拌和离心等工位必须全程覆盖头发。",
            "头罩污染后及时更换，防止二次污染。",
        ],
    },
    "火焰失控": {
        "severity": "极高危",
        "hazard": "可能引发火灾、爆炸或连锁事故，威胁人员与设备安全。",
        "standard": "危险化学品实验室安全通则",
        "correct_method": [
            "立即停止加热并切断可燃气源。",
            "按预案使用合适灭火器材或防火毯处置。",
            "同时组织人员撤离并上报现场负责人。",
        ],
    },
    "化学品泄漏": {
        "severity": "极高危",
        "hazard": "有毒有害物质扩散，可能造成中毒、腐蚀、火灾或环境污染。",
        "standard": "危险化学品泄漏应急处置规范",
        "correct_method": [
            "立即隔离泄漏区域，阻止无关人员靠近。",
            "佩戴相应防护装备后使用吸附材料进行围堵处置。",
            "按化学品类别进行废弃物收集并上报登记。",
        ],
    },
}

USERS = {
    "admin001": {
        "userId": "admin-001",
        "username": "admin001",
        "password": "123456",
        "role": "admin",
        "avatar": "",
    },
    "user001": {
        "userId": DEFAULT_RECORD_USER_ID,
        "username": DEFAULT_RECORD_USERNAME,
        "password": "123456",
        "role": "user",
        "avatar": "",
    },
}

app = Flask(__name__)


@dataclass
class AuthUser:
    user_id: str
    username: str
    role: str
    avatar: str = ""


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_event_timestamp(value):
    if not value:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def to_iso8601(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    parsed = parse_event_timestamp(str(value))
    if parsed is not None:
        return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return now_iso()


def parse_client_date(value):
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def pick_video_root():
    for path in VIDEO_DIR_CANDIDATES:
        if path.exists():
            return path
    return VIDEO_DIR_CANDIDATES[0]


def api_ok(data=None, message="success"):
    return jsonify({"code": 200, "message": message, "data": data})


def api_error(code, message, http_status):
    return jsonify({"code": code, "message": message, "data": None}), http_status


@app.route("/", methods=["GET"])
def index():
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Harmony API Server</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #182230;
    }}
    main {{
      max-width: 760px;
      margin: 64px auto;
      padding: 32px;
      background: #fff;
      border: 1px solid #d9e1ec;
      border-radius: 10px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, .08);
    }}
    h1 {{ margin: 0 0 12px; font-size: 28px; }}
    p {{ line-height: 1.7; }}
    code {{
      display: inline-block;
      padding: 3px 6px;
      background: #eef3f8;
      border-radius: 5px;
    }}
    ul {{ line-height: 2; }}
    a {{ color: #0f5ea8; }}
    footer {{
      margin-top: 28px;
      font-size: 12px;
      color: #4b5d73;
      text-align: center;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Harmony API Server is running</h1>
    <p>板卡业务服务已经启动，当前端口是 <code>{PORT}</code>。</p>
    <ul>
      <li><a href="/health">/health</a>：服务健康检查</li>
      <li><a href="/stream-test">/stream-test</a>：浏览器实时视频流测试页</li>
      <li><a href="/analysis-test">/analysis-test</a>：浏览器图片分析列表测试页</li>
      <li><code>POST /api/auth/login</code>：登录接口</li>
      <li><code>GET /api/analysis/images</code>：图片分析列表，需要 token</li>
      <li><code>GET /api/media/videos</code>：科普视频列表，需要 token</li>
    </ul>
    <footer>嘉瑞好帅</footer>
  </main>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/health", methods=["GET"])
def health():
    return api_ok({
        "status": "ok",
        "service": "harmony-api",
        "host": HOST,
        "port": PORT,
        "time": now_iso(),
    })


def build_token_payload(user, expire_at):
    return {
        "userId": user["userId"],
        "username": user["username"],
        "role": user["role"],
        "avatar": user.get("avatar", ""),
        "exp": expire_at,
    }


def encode_token(payload):
    body = base64.urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = hmac.new(TOKEN_SECRET.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def decode_token(token):
    try:
        body, signature = token.split(".", 1)
    except ValueError:
        raise ValueError("invalid token")

    expected = hmac.new(TOKEN_SECRET.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid token")

    padded = body + "=" * (-len(body) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("token expired")
    return payload


def extract_bearer_token():
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip()


def require_auth(required_role: Optional[str] = None):
    token = extract_bearer_token()
    if not token:
        return None, api_error(401, "missing bearer token", 401)

    try:
        payload = decode_token(token)
    except ValueError as exc:
        return None, api_error(401, str(exc), 401)

    user = AuthUser(
        user_id=str(payload["userId"]),
        username=str(payload["username"]),
        role=str(payload["role"]),
        avatar=str(payload.get("avatar", "")),
    )
    if required_role and user.role != required_role:
        return None, api_error(403, "forbidden", 403)
    return user, None


def normalize_violation_name(name):
    if not name:
        return "未知违规"
    aliases = {
        "未戴护目镜": "未佩戴护目镜",
        "酒精灯": "明火",
        "火焰": "明火",
    }
    return aliases.get(name, name)


def unique_violation_names(names):
    result = []
    seen = set()
    for name in names or []:
        normalized = normalize_violation_name(str(name))
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def get_analysis_metadata(class_name):
    normalized = normalize_violation_name(class_name)
    if normalized in VIOLATION_METADATA:
        return VIOLATION_METADATA[normalized]
    return {
        "severity": "中危",
        "hazard": f"{normalized}可能导致实验风险升高，需要及时整改。",
        "standard": "实验室安全管理规范",
        "correct_method": [
            "立即停止当前高风险操作。",
            "按照实验室个人防护和操作规范完成整改。",
            "由现场负责人复核后再恢复实验。",
        ],
    }


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_violation_events():
    if not EVENT_LOG_PATH.exists():
        return []

    events = []
    with EVENT_LOG_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
    return events


def append_violation_event(event):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def existing_event_keys():
    keys = set()
    for event in load_violation_events():
        image_path = str(event.get("image_path") or "")
        if image_path:
            keys.add(image_path)
        event_id = str(event.get("id") or event.get("event_id") or "")
        if event_id:
            keys.add(event_id)
    return keys


def event_id_from_path(image_path, fallback_index):
    if image_path:
        return Path(image_path).stem
    return f"event-{fallback_index:06d}"


def guess_result_image_path(event):
    result_image_path = Path(event.get("result_image_path", ""))
    if result_image_path.is_file():
        return result_image_path.resolve()

    image_path = Path(event.get("image_path", ""))
    if image_path.is_file():
        return image_path.resolve()
    return None


def guess_original_image_path(result_path: Optional[Path], event=None):
    if isinstance(event, dict):
        original_image_path = Path(event.get("original_image_path", ""))
        if original_image_path.is_file():
            return original_image_path.resolve()

    if result_path is None:
        return None

    stem = result_path.stem
    suffix = result_path.suffix
    for marker in ("_violation", "_detected", "-detected"):
        if marker in stem:
            candidate = result_path.with_name(stem.replace(marker, "") + suffix)
            if candidate.is_file():
                return candidate.resolve()
    return result_path.resolve() if result_path.is_file() else None


def build_media_url(kind, path):
    encoded = quote(str(path.resolve()))
    if MEDIA_BASE_URL:
        base_url = MEDIA_BASE_URL
    elif PUBLIC_BASE_URL:
        base_url = PUBLIC_BASE_URL
    else:
        host = request.host if has_request_context() else f"127.0.0.1:{PORT}"
        scheme = request.scheme if has_request_context() else "http"
        base_url = f"{scheme}://{host}"
    return f"{base_url}/media/{kind}?path={encoded}"


def build_analysis_item(event, index):
    result_path = guess_result_image_path(event)
    original_path = guess_original_image_path(result_path, event)

    violation_details = event.get("violation_details", [])
    if not isinstance(violation_details, list):
        violation_details = []
    violation_details = [
        {
            **item,
            "className": normalize_violation_name(str(item.get("className", ""))),
        }
        for item in violation_details
        if isinstance(item, dict)
        and normalize_violation_name(str(item.get("className", ""))) in ALLOWED_ANALYSIS_VIOLATIONS
    ]
    violation_names = unique_violation_names(event.get("violation_names", []))
    if not violation_names:
        violation_names = unique_violation_names(
            item.get("className")
            for item in violation_details
            if isinstance(item, dict)
        )
    violation_names = [name for name in violation_names if name in ALLOWED_ANALYSIS_VIOLATIONS]
    if not violation_names:
        return None
    class_name = "、".join(violation_names)
    metadata = get_analysis_metadata(violation_names[0])

    item_id = event_id_from_path(event.get("image_path"), index)
    confidence_values = [
        safe_float(item.get("confidence"), 0.0)
        for item in violation_details
        if isinstance(item, dict)
    ]
    confidence = max(confidence_values) if confidence_values else None
    unix_time = safe_float(event.get("unix_time"), 0.0)
    create_time = to_iso8601(unix_time or event.get("timestamp"))

    user_id = str(event.get("user_id") or DEFAULT_RECORD_USER_ID)
    username = str(event.get("username") or DEFAULT_RECORD_USERNAME)

    return {
        "id": item_id,
        "imageUrl": build_media_url("image", original_path) if original_path else "",
        "resultImageUrl": build_media_url("image", result_path) if result_path else "",
        "className": class_name,
        "classNames": violation_names,
        "confidence": confidence,
        "violationDetails": violation_details,
        "createTime": create_time,
        "userId": user_id,
        "username": username,
        "aiAnalysis": {
            "hazard": metadata["hazard"],
            "severity": metadata["severity"],
            "standard": metadata["standard"],
        },
        "correctMethod": metadata["correct_method"],
        "alertText": event.get("alert_text", ""),
        "recordType": "analysis",
        "status": str(event.get("status") or "pending"),
        "sourceImagePath": str(original_path) if original_path else "",
        "resultImagePath": str(result_path) if result_path else "",
    }


def load_analysis_records():
    items = [
        item
        for item in (
            build_analysis_item(event, index)
            for index, event in enumerate(load_violation_events(), start=1)
        )
        if item is not None
    ]
    items.sort(key=lambda item: item["createTime"], reverse=True)
    return items


def clear_analysis_history():
    deleted_files = 0
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENT_LOG_PATH.write_text("", encoding="utf-8")

    if VIOLATIONS_DIR.exists():
        for path in VIOLATIONS_DIR.iterdir():
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    deleted_files += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    deleted_files += 1
            except OSError:
                continue
    else:
        VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

    return deleted_files


def paginate(items, page, page_size):
    total = len(items)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return total, items[start:end]


def parse_paging_args():
    page = request.args.get("page", "1")
    page_size = request.args.get("pageSize", "20")
    try:
        page = max(1, int(page))
        page_size = max(1, min(100, int(page_size)))
    except ValueError:
        raise ValueError("page and pageSize must be integers")
    return page, page_size


def filter_items_by_date(items, start_key, end_key, time_field):
    start_date = parse_client_date(request.args.get(start_key))
    end_date = parse_client_date(request.args.get(end_key))
    filtered = []
    for item in items:
        item_time = parse_client_date(str(item.get(time_field, "")))
        if item_time is None:
            continue
        if start_date and item_time < start_date:
            continue
        if end_date and item_time > end_date:
            continue
        filtered.append(item)
    return filtered


def build_record_item(analysis_item):
    return {
        "id": analysis_item["id"],
        "userId": analysis_item["userId"],
        "username": analysis_item["username"],
        "recordType": analysis_item["recordType"],
        "eventTime": analysis_item["createTime"],
        "className": analysis_item["className"],
        "thumbnailUrl": analysis_item["resultImageUrl"] or analysis_item["imageUrl"],
        "status": analysis_item["status"],
    }


def detect_duration_seconds(video_path):
    try:
        import cv2
    except ImportError:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0 or frames <= 0:
        return 0
    return int(frames / fps)


def build_video_item(video_path, index):
    stats = video_path.stat()
    video_id = f"video-{index:04d}"
    title = video_path.stem
    cover_path = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = video_path.with_suffix(ext)
        if candidate.is_file():
            cover_path = candidate.resolve()
            break

    description = f"实验室安全科普视频：{title}"
    return {
        "id": video_id,
        "title": title,
        "coverUrl": build_media_url("image", cover_path) if cover_path else "",
        "videoUrl": build_media_url("video", video_path),
        "duration": detect_duration_seconds(video_path),
        "description": description,
        "updatedAt": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def load_videos():
    root = pick_video_root()
    root.mkdir(parents=True, exist_ok=True)
    items = []
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    for index, path in enumerate(sorted(root.iterdir()), start=1):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        items.append(build_video_item(path, index))
    return items


def call_deepseek_analysis(class_name, alert_text, image_id):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    prompt = (
        "你是化工实验室安全分析助手。"
        "请基于给定违规类型输出JSON，不要输出markdown。"
        '字段必须包含 hazard, severity, standard, correctMethod。'
        "其中 correctMethod 是长度为3的中文字符串数组。"
        f"违规类型：{class_name}\n"
        f"告警内容：{alert_text}\n"
        f"记录ID：{image_id}\n"
    )
    response = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        messages=[
            {"role": "system", "content": "你输出严格JSON。"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=256,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    role = str(payload.get("role", "")).strip()

    user = USERS.get(username)
    if user is None or user["password"] != password or user["role"] != role:
        return api_error(401, "invalid username, password, or role", 401)

    expire_at = int(time.time()) + TOKEN_EXPIRE_SECONDS
    token = encode_token(build_token_payload(user, expire_at))
    return api_ok(
        {
            "token": token,
            "userId": user["userId"],
            "username": user["username"],
            "role": user["role"],
            "avatar": user.get("avatar", ""),
            "expireAt": to_iso8601(expire_at),
        }
    )


@app.route("/api/analysis/images", methods=["GET"])
def analysis_images():
    user, error_resp = require_auth()
    if error_resp:
        return error_resp

    try:
        page, page_size = parse_paging_args()
    except ValueError as exc:
        return api_error(400, str(exc), 400)

    items = load_analysis_records()
    class_name = request.args.get("className", "").strip()
    if class_name:
        normalized_filter = normalize_violation_name(class_name)
        items = [item for item in items if normalized_filter in item.get("classNames", [item["className"]])]
    items = filter_items_by_date(items, "startDate", "endDate", "createTime")

    if user.role != "admin":
        items = [item for item in items if item["userId"] == user.user_id]

    total, page_items = paginate(items, page, page_size)
    list_items = [
        {
            "id": item["id"],
            "imageUrl": item["imageUrl"],
            "resultImageUrl": item["resultImageUrl"],
            "className": item["className"],
            "classNames": item["classNames"],
            "confidence": item["confidence"],
            "violationDetails": item["violationDetails"],
            "createTime": item["createTime"],
            "userId": item["userId"],
            "username": item["username"],
        }
        for item in page_items
    ]
    return api_ok(
        {
            "total": total,
            "page": page,
            "pageSize": page_size,
            "list": list_items,
        }
    )


@app.route("/api/analysis/images/clear", methods=["POST", "OPTIONS"])
def analysis_images_clear():
    if request.method == "OPTIONS":
        return ("", 204)

    user, error_resp = require_auth(required_role="admin")
    if error_resp:
        return error_resp

    deleted_files = clear_analysis_history()
    return api_ok({"cleared": True, "deletedFiles": deleted_files})


@app.route("/api/analysis/events/sync", methods=["POST", "OPTIONS"])
def analysis_events_sync():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    events = payload.get("events")
    if not isinstance(events, list):
        return api_error(400, "events must be a list", 400)

    seen = existing_event_keys()
    saved = 0
    skipped = 0
    for event in events:
        if not isinstance(event, dict):
            skipped += 1
            continue

        image_path = str(event.get("image_path") or "")
        event_id = str(event.get("id") or event.get("event_id") or "")
        key = image_path or event_id
        if key and key in seen:
            skipped += 1
            continue

        append_violation_event(event)
        saved += 1
        if key:
            seen.add(key)

    return api_ok({"saved": saved, "skipped": skipped, "received": len(events)})


@app.route("/api/analysis/images/<image_id>", methods=["GET"])
def analysis_image_detail(image_id):
    user, error_resp = require_auth()
    if error_resp:
        return error_resp

    for item in load_analysis_records():
        if item["id"] != image_id:
            continue
        if user.role != "admin" and item["userId"] != user.user_id:
            return api_error(403, "forbidden", 403)
        return api_ok(item)
    return api_error(404, "image record not found", 404)


@app.route("/api/analysis/ai-analyze", methods=["POST", "OPTIONS"])
def analysis_ai_analyze():
    if request.method == "OPTIONS":
        return ("", 204)

    user, error_resp = require_auth()
    if error_resp:
        return error_resp

    payload = request.get_json(silent=True) or {}
    image_id = str(payload.get("id") or payload.get("imageId") or "").strip()
    if not image_id:
        return api_error(400, "id or imageId is required", 400)

    items = load_analysis_records()
    target = next((item for item in items if item["id"] == image_id), None)
    if target is None:
        return api_error(404, "image record not found", 404)
    if user.role != "admin" and target["userId"] != user.user_id:
        return api_error(403, "forbidden", 403)

    ai_result = call_deepseek_analysis(target["className"], target["alertText"], target["id"])
    if ai_result:
        hazard = str(ai_result.get("hazard", target["aiAnalysis"]["hazard"]))
        severity = str(ai_result.get("severity", target["aiAnalysis"]["severity"]))
        standard = str(ai_result.get("standard", target["aiAnalysis"]["standard"]))
        correct_method = ai_result.get("correctMethod", target["correctMethod"])
        if not isinstance(correct_method, list) or not correct_method:
            correct_method = target["correctMethod"]
    else:
        hazard = target["aiAnalysis"]["hazard"]
        severity = target["aiAnalysis"]["severity"]
        standard = target["aiAnalysis"]["standard"]
        correct_method = target["correctMethod"]

    return api_ok(
        {
            "id": target["id"],
            "className": target["className"],
            "classNames": target["classNames"],
            "confidence": target["confidence"],
            "createTime": target["createTime"],
            "imageUrl": target["imageUrl"],
            "resultImageUrl": target["resultImageUrl"],
            "aiAnalysis": {
                "hazard": hazard,
                "severity": severity,
                "standard": standard,
            },
            "correctMethod": correct_method,
        }
    )


@app.route("/api/media/videos", methods=["GET"])
def media_videos():
    user, error_resp = require_auth()
    if error_resp:
        return error_resp

    try:
        page, page_size = parse_paging_args()
    except ValueError as exc:
        return api_error(400, str(exc), 400)

    videos = load_videos()
    total, page_items = paginate(videos, page, page_size)
    return api_ok(
        {
            "total": total,
            "page": page,
            "pageSize": page_size,
            "list": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "coverUrl": item["coverUrl"],
                    "videoUrl": item["videoUrl"],
                    "duration": item["duration"],
                    "description": item["description"],
                }
                for item in page_items
            ],
        }
    )


@app.route("/api/records/my", methods=["GET"])
def records_my():
    user, error_resp = require_auth()
    if error_resp:
        return error_resp

    try:
        page, page_size = parse_paging_args()
    except ValueError as exc:
        return api_error(400, str(exc), 400)

    record_type = request.args.get("recordType", "").strip()
    records = [build_record_item(item) for item in load_analysis_records()]
    records = [item for item in records if item["userId"] == user.user_id]
    if record_type:
        records = [item for item in records if item["recordType"] == record_type]
    records = filter_items_by_date(records, "startDate", "endDate", "eventTime")

    total, page_items = paginate(records, page, page_size)
    return api_ok({"total": total, "page": page, "pageSize": page_size, "list": page_items})


@app.route("/api/records/all", methods=["GET"])
def records_all():
    user, error_resp = require_auth(required_role="admin")
    if error_resp:
        return error_resp

    try:
        page, page_size = parse_paging_args()
    except ValueError as exc:
        return api_error(400, str(exc), 400)

    user_id = request.args.get("userId", "").strip()
    records = [build_record_item(item) for item in load_analysis_records()]
    if user_id:
        records = [item for item in records if item["userId"] == user_id]
    records = filter_items_by_date(records, "startDate", "endDate", "eventTime")

    total, page_items = paginate(records, page, page_size)
    return api_ok({"total": total, "page": page, "pageSize": page_size, "list": page_items})


@app.route("/media/<kind>", methods=["GET"])
def serve_media(kind):
    path_arg = request.args.get("path", "")
    if not path_arg:
        return api_error(400, "path is required", 400)

    try:
        requested = Path(path_arg).resolve()
    except OSError:
        return api_error(400, "invalid path", 400)

    allowed_roots = [
        VIOLATIONS_DIR.resolve(),
        (BASE_DIR / "image").resolve(),
        pick_video_root().resolve(),
    ]
    if not any(root == requested or root in requested.parents for root in allowed_roots):
        return api_error(403, "path is outside allowed roots", 403)
    if not requested.exists() or not requested.is_file():
        return api_error(404, "media file not found", 404)

    mime_type, _ = mimetypes.guess_type(str(requested))
    return send_file(str(requested), mimetype=mime_type or ("video/mp4" if kind == "video" else "image/jpeg"))


def print_startup_summary():
    print("[INFO] Harmony API server")
    print(f"[INFO] listening on http://127.0.0.1:{PORT}")
    if PUBLIC_BASE_URL:
        print(f"[INFO] public base url: {PUBLIC_BASE_URL}")
    if MEDIA_BASE_URL:
        print(f"[INFO] media base url : {MEDIA_BASE_URL}")
    print(f"[INFO] event log: {EVENT_LOG_PATH}")
    print(f"[INFO] violation images: {VIOLATIONS_DIR}")
    print(f"[INFO] video root: {pick_video_root()}")
    print("[INFO] demo admin: admin001 / 123456")
    print("[INFO] demo user : user001 / 123456")


if __name__ == "__main__":
    print_startup_summary()
    app.run(host=HOST, port=PORT, debug=False)
