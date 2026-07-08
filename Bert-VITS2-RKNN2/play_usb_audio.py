import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DEVICE = "plughw:4,0"


def require_command(command_name):
    command_path = shutil.which(command_name)
    if not command_path:
        raise RuntimeError(f"缺少命令: {command_name}")
    return command_path


def prepare_usb_playback(input_path, output_path, volume):
    ffmpeg = require_command("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-filter:a",
        f"volume={volume}",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"音频转换失败，退出码: {completed.returncode}")


def play_prepared_wav(input_path, device):
    aplay = require_command("aplay")
    command = [aplay, "-D", device, str(input_path)]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"音频播放失败，退出码: {completed.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(description="将 WAV 转换为 USB 扬声器兼容格式并播放。")
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="ALSA 播放设备，默认 plughw:4,0",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="软件音量倍率，1.0 为原始音量，0.5 为半音量，2.0 为双倍音量",
    )
    parser.add_argument(
        "--prepared-output",
        help="转换后的 WAV 输出路径，默认写到输入文件同目录下 *_usb.wav",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"找不到输入文件: {input_path}")
    if args.volume <= 0:
        raise RuntimeError("volume 必须大于 0")

    if args.prepared_output:
        prepared_output = Path(args.prepared_output).expanduser().resolve()
    else:
        prepared_output = input_path.with_name(f"{input_path.stem}_usb.wav")

    print(f"输入文件: {input_path}")
    print(f"转换输出: {prepared_output}")
    print(f"播放设备: {args.device}")
    print(f"软件音量倍率: {args.volume}")
    sys.stdout.flush()

    prepare_usb_playback(input_path, prepared_output, args.volume)
    play_prepared_wav(prepared_output, args.device)
    print("播放完成。")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
