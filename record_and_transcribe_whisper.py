import argparse
import os
import subprocess
import sys
from pathlib import Path

from microphone_recorder import choose_device, probe_recording_params


SCRIPT_DIR = Path(__file__).resolve().parent
WHISPER_DIR = SCRIPT_DIR / "whisper"
WHISPER_PY = WHISPER_DIR / "python" / "whisper.py"
ENCODER_MODEL = WHISPER_DIR / "model" / "whisper_encoder_base_20s.rknn"
DECODER_MODEL = WHISPER_DIR / "model" / "whisper_decoder_base_20s.rknn"
DEFAULT_AUDIO = SCRIPT_DIR / "results" / "webcam_mic_whisper.wav"
DEFAULT_TRANSCRIPT = SCRIPT_DIR / "results" / "webcam_mic_whisper.txt"


def record_audio(device, duration, rate, channels, fmt, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = probe_recording_params(device, rate, channels, fmt)

    command = [
        "arecord",
        "-D",
        params["device"],
        "-d",
        str(duration),
        "-f",
        params["format"],
        "-r",
        str(params["rate"]),
        "-c",
        str(params["channels"]),
        str(output_path),
    ]

    print(f"开始录音，设备: {params['device']}")
    print(
        "录音参数:",
        f"format={params['format']}",
        f"rate={params['rate']}",
        f"channels={params['channels']}",
    )
    print("执行命令:", " ".join(command))

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError("录音失败。")


def run_whisper(audio_path, task, target, device_id):
    if not WHISPER_PY.is_file():
        raise RuntimeError(f"未找到 whisper 脚本: {WHISPER_PY}")
    if not ENCODER_MODEL.is_file():
        raise RuntimeError(f"未找到 encoder 模型: {ENCODER_MODEL}")
    if not DECODER_MODEL.is_file():
        raise RuntimeError(f"未找到 decoder 模型: {DECODER_MODEL}")

    python_bin = os.environ.get("WHISPER_PYTHON", "/root/miniconda3/envs/rknn/bin/python")
    if not Path(python_bin).is_file():
        raise RuntimeError(f"未找到用于运行 whisper 的 Python: {python_bin}")

    command = [
        python_bin,
        str(WHISPER_PY),
        "--encoder_model_path",
        str(ENCODER_MODEL),
        "--decoder_model_path",
        str(DECODER_MODEL),
        "--task",
        task,
        "--audio_path",
        str(audio_path),
        "--target",
        target,
    ]
    if device_id:
        command.extend(["--device_id", device_id])

    print("执行 Whisper 命令:", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(WHISPER_PY.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        raise RuntimeError(f"Whisper 转写失败。\n{details}")

    output_text = completed.stdout.strip()
    for line in reversed(output_text.splitlines()):
        if line.startswith("Whisper output:"):
            return line.split("Whisper output:", 1)[1].strip(), output_text

    raise RuntimeError(f"Whisper 已运行，但未解析到结果。\n{output_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="录音并调用本地 RKNN Whisper 模型转文字。")
    parser.add_argument("--device", default="hw:3,0", help="录音设备")
    parser.add_argument("--match", help="按关键字匹配录音设备")
    parser.add_argument("--duration", type=int, default=5, help="录音时长，单位秒")
    parser.add_argument("--output", default=str(DEFAULT_AUDIO), help="WAV 输出路径")
    parser.add_argument("--transcript", default=str(DEFAULT_TRANSCRIPT), help="文本输出路径")
    parser.add_argument("--rate", type=int, default=16000, help="期望采样率")
    parser.add_argument("--channels", type=int, default=1, help="期望声道数")
    parser.add_argument("--format", default="S16_LE", help="采样格式")
    parser.add_argument("--task", choices=["zh", "en"], default="zh", help="识别语言")
    parser.add_argument("--target", default="rk3588", help="RKNN target")
    parser.add_argument("--device-id", default=None, help="RKNN device id")
    parser.add_argument("--record-only", action="store_true", help="只录音，不转写")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    transcript_path = Path(args.transcript).expanduser().resolve()

    try:
        device = choose_device(args.device, args.match)
        record_audio(
            device=device,
            duration=args.duration,
            rate=args.rate,
            channels=args.channels,
            fmt=args.format,
            output_path=output_path,
        )
        print(f"录音完成: {output_path}")

        if args.record_only:
            return

        result, raw_output = run_whisper(
            audio_path=output_path,
            task=args.task,
            target=args.target,
            device_id=args.device_id,
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(result + "\n", encoding="utf-8")
        print(f"转写完成: {transcript_path}")
        print(result)
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
