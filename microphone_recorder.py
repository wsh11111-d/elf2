import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_command(command):
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def get_arecord_hw_devices():
    code, stdout, stderr = run_command(["arecord", "-l"])
    text = "\n".join(part for part in [stdout, stderr] if part).strip()
    if code != 0 and not text:
        raise RuntimeError("无法执行 arecord -l，请确认系统已安装 ALSA 工具。")

    devices = []
    pattern = re.compile(
        r"card\s+(?P<card>\d+):\s+(?P<card_id>[^\s]+)\s+\[(?P<card_name>.*?)\],\s+"
        r"device\s+(?P<device>\d+):\s+(?P<device_name>.*?)\s+\[(?P<device_desc>.*?)\]"
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        card = match.group("card")
        device = match.group("device")
        devices.append(
            {
                "id": f"hw:{card},{device}",
                "card": card,
                "device": device,
                "label": (
                    f"{match.group('card_name')} / "
                    f"{match.group('device_name')} / "
                    f"{match.group('device_desc')}"
                ),
                "raw": line.strip(),
            }
        )
    return devices, text


def get_arecord_pcm_devices():
    code, stdout, stderr = run_command(["arecord", "-L"])
    text = "\n".join(part for part in [stdout, stderr] if part).strip()
    if code != 0 and not text:
        raise RuntimeError("无法执行 arecord -L，请确认系统已安装 ALSA 工具。")
    return text


def print_devices():
    hw_devices, hw_text = get_arecord_hw_devices()
    pcm_text = get_arecord_pcm_devices()

    print("=== `arecord -l` 录音硬件设备 ===")
    if hw_devices:
        for device in hw_devices:
            print(f"{device['id']}: {device['label']}")
    else:
        print(hw_text or "没有检测到录音硬件设备。")

    print("\n=== `arecord -L` PCM 设备 ===")
    print(pcm_text or "没有检测到 PCM 设备。")


def choose_device(device_arg, match_text):
    if device_arg:
        return device_arg

    hw_devices, hw_text = get_arecord_hw_devices()
    if not hw_devices:
        raise RuntimeError(
            "没有检测到录音设备。\n"
            f"`arecord -l` 输出：\n{hw_text or '空'}"
        )

    if match_text:
        keyword = match_text.lower()
        for device in hw_devices:
            haystack = f"{device['id']} {device['label']} {device['raw']}".lower()
            if keyword in haystack:
                return device["id"]
        raise RuntimeError(
            f"没有找到包含关键字 `{match_text}` 的录音设备，请先运行 `--list-devices` 查看。"
        )

    return hw_devices[0]["id"]


def normalize_device_variants(device_name):
    variants = [device_name]
    if device_name.startswith("hw:") and not device_name.startswith("plughw:"):
        variants.append(device_name.replace("hw:", "plughw:", 1))
    return variants


def probe_recording_params(device_name, requested_rate, requested_channels, requested_format):
    candidate_rates = []
    for rate in [requested_rate, 48000, 44100, 16000, 8000]:
        if rate not in candidate_rates:
            candidate_rates.append(rate)

    candidate_channels = []
    for channels in [requested_channels, 2, 1]:
        if channels not in candidate_channels:
            candidate_channels.append(channels)

    candidate_formats = []
    for fmt in [requested_format, "S16_LE"]:
        if fmt not in candidate_formats:
            candidate_formats.append(fmt)

    last_error = ""
    for device_variant in normalize_device_variants(device_name):
        for channels in candidate_channels:
            for rate in candidate_rates:
                for fmt in candidate_formats:
                    command = [
                        "arecord",
                        "-D",
                        device_variant,
                        "--dump-hw-params",
                        "-d",
                        "1",
                        "-f",
                        fmt,
                        "-r",
                        str(rate),
                        "-c",
                        str(channels),
                        "/tmp/codex_audio_probe.wav",
                    ]
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if completed.returncode == 0:
                        return {
                            "device": device_variant,
                            "rate": rate,
                            "channels": channels,
                            "format": fmt,
                        }
                    last_error = "\n".join(
                        part for part in [completed.stdout, completed.stderr] if part
                    ).strip()
    raise RuntimeError(
        "没有探测到可用录音参数。\n"
        f"最后一次错误信息：\n{last_error or '空'}"
    )


def record_audio(args):
    selected_device = choose_device(args.device, args.match)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_params = probe_recording_params(
        selected_device,
        args.rate,
        args.channels,
        args.format,
    )

    command = [
        "arecord",
        "-D",
        selected_params["device"],
        "-d",
        str(args.duration),
        "-f",
        selected_params["format"],
        "-r",
        str(selected_params["rate"]),
        "-c",
        str(selected_params["channels"]),
        str(output_path),
    ]

    print(f"开始录音，设备: {selected_params['device']}")
    print(f"输出文件: {output_path}")
    print(
        "录音参数:",
        f"format={selected_params['format']}",
        f"rate={selected_params['rate']}",
        f"channels={selected_params['channels']}",
    )
    print("执行命令:", " ".join(command))

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "录音失败。通常是设备号不对、摄像头麦克风未被系统识别，或者当前账号没有录音权限。"
        )

    print("录音完成。")


def build_parser():
    parser = argparse.ArgumentParser(description="使用 ALSA/arecord 调用摄像头麦克风录音。")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出当前系统可用的录音设备",
    )
    parser.add_argument(
        "--device",
        help="显式指定 ALSA 设备，例如 hw:1,0 或 plughw:1,0",
    )
    parser.add_argument(
        "--match",
        help="按关键字匹配设备，例如 usb、webcam、camera、UVC",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="录音时长，单位秒",
    )
    parser.add_argument(
        "--output",
        default="demo/results/mic_test.wav",
        help="输出 WAV 文件路径",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="采样率",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="声道数",
    )
    parser.add_argument(
        "--format",
        default="S16_LE",
        help="采样格式，默认 S16_LE",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.list_devices:
            print_devices()
            return
        record_audio(args)
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
