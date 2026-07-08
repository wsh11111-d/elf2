import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

from microphone_recorder import choose_device, print_devices
from record_and_transcribe_whisper import record_audio, run_whisper


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_AUDIO = RESULTS_DIR / "chem_lab_voice_input.wav"
DEFAULT_TRANSCRIPT = RESULTS_DIR / "chem_lab_transcript.txt"
DEFAULT_SUMMARY = RESULTS_DIR / "chem_lab_summary.txt"
DEFAULT_TTS_WAV = RESULTS_DIR / "chem_lab_summary.wav"
RKNN_PYTHON = Path("/root/miniconda3/envs/rknn/bin/python")
TTS_DIR = Path("/root/wsh/Bert-VITS2-RKNN2")
TTS_SCRIPT = TTS_DIR / "rknn_run.py"
PLAY_SCRIPT = TTS_DIR / "play_usb_audio.py"


def build_parser():
    parser = argparse.ArgumentParser(
        description="录音 -> Whisper 转写 -> DeepSeek 总结 -> Bert-VITS2 播报的化学实验语音辅助流程。"
    )
    parser.add_argument("--list-devices", action="store_true", help="列出当前可用录音设备")
    parser.add_argument("--device", default="hw:3,0", help="录音设备，默认 hw:3,0")
    parser.add_argument("--match", help="按关键字匹配录音设备，例如 usb、camera、UVC")
    parser.add_argument("--duration", type=int, default=10, help="录音时长，默认 10 秒")
    parser.add_argument("--rate", type=int, default=16000, help="录音采样率")
    parser.add_argument("--channels", type=int, default=1, help="录音声道数")
    parser.add_argument("--format", default="S16_LE", help="录音采样格式")
    parser.add_argument("--task", choices=["zh", "en"], default="zh", help="Whisper 识别语言")
    parser.add_argument("--target", default="rk3588", help="Whisper RKNN target")
    parser.add_argument("--device-id", default=None, help="Whisper device id")
    parser.add_argument("--audio-output", default=str(DEFAULT_AUDIO), help="录音输出路径")
    parser.add_argument("--transcript-output", default=str(DEFAULT_TRANSCRIPT), help="转写文本输出路径")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY), help="大模型总结文本输出路径")
    parser.add_argument("--tts-output", default=str(DEFAULT_TTS_WAV), help="TTS 输出音频路径")
    parser.add_argument("--model", default="deepseek-v4-pro", help="DeepSeek 模型名")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek API base_url")
    parser.add_argument("--speaker-device", default="plughw:4,0", help="USB 喇叭 ALSA 设备")
    parser.add_argument("--speaker-volume", type=float, default=1.0, help="喇叭软件音量倍率")
    parser.add_argument("--summary-max-chars", type=int, default=100, help="口播文案最大字数")
    parser.add_argument("--skip-playback", action="store_true", help="只生成音频，不实际播放")
    return parser


def require_api_key():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError(
        "缺少 DEEPSEEK_API_KEY。请先在当前终端执行:\n"
        "export DEEPSEEK_API_KEY='your_api_key_here'"
    )


def clean_summary_text(text, max_chars):
    text = re.sub(r"\s+", "", text.strip())
    text = text.strip("“”\"'`")
    text = text.replace("总结：", "").replace("提醒：", "")
    if len(text) <= max_chars:
        return text

    cut_candidates = "。；！？，,"
    for index in range(max_chars, 0, -1):
        if text[index - 1] in cut_candidates:
            return text[:index]
    return text[:max_chars]


def summarize_with_deepseek(transcript, model, base_url, max_chars):
    client = OpenAI(
        api_key=require_api_key(),
        base_url=base_url,
    )

    system_prompt = (
        "你是化学实验室行为规范语音助手。"
    )
    user_prompt = (
        "请根据下面的语音转写内容，总结实验中的易错点和规范提醒。"
        f"要求：只输出一段中文，不超过{max_chars}字，不要标题，不要分点，不要解释，"
        "语气自然，适合直接播报。\n\n"
        f"语音转写内容：{transcript or '未识别到清晰语音，请给出通用化学实验规范提醒。'}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort="high",
        max_tokens=180,
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = response.choices[0].message.content or ""
    result = clean_summary_text(content, max_chars)
    if not result:
        raise RuntimeError("DeepSeek 返回了空结果。")
    return result


def run_tts(text, output_path):
    if not RKNN_PYTHON.is_file():
        raise RuntimeError(f"未找到 RKNN Python: {RKNN_PYTHON}")
    if not TTS_SCRIPT.is_file():
        raise RuntimeError(f"未找到 TTS 脚本: {TTS_SCRIPT}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(RKNN_PYTHON),
        str(TTS_SCRIPT),
        "--text",
        text,
        "--output",
        str(output_path),
    ]
    print("执行 TTS 命令:", " ".join(command))
    completed = subprocess.run(command, cwd=str(TTS_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"TTS 生成失败，退出码: {completed.returncode}")


def play_audio(audio_path, speaker_device, speaker_volume):
    if not RKNN_PYTHON.is_file():
        raise RuntimeError(f"未找到 RKNN Python: {RKNN_PYTHON}")
    if not PLAY_SCRIPT.is_file():
        raise RuntimeError(f"未找到播放脚本: {PLAY_SCRIPT}")

    command = [
        str(RKNN_PYTHON),
        str(PLAY_SCRIPT),
        str(audio_path),
        "--device",
        speaker_device,
        "--volume",
        str(speaker_volume),
    ]
    print("执行播放命令:", " ".join(command))
    completed = subprocess.run(command, cwd=str(TTS_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"音频播放失败，退出码: {completed.returncode}")


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        print_devices()
        return

    audio_output = Path(args.audio_output).expanduser().resolve()
    transcript_output = Path(args.transcript_output).expanduser().resolve()
    summary_output = Path(args.summary_output).expanduser().resolve()
    tts_output = Path(args.tts_output).expanduser().resolve()

    try:
        print("步骤 1/4: 开始录音")
        device = choose_device(args.device, args.match)
        record_audio(
            device=device,
            duration=args.duration,
            rate=args.rate,
            channels=args.channels,
            fmt=args.format,
            output_path=audio_output,
        )
        print(f"录音完成: {audio_output}")

        print("步骤 2/4: Whisper 转写")
        transcript, _ = run_whisper(
            audio_path=audio_output,
            task=args.task,
            target=args.target,
            device_id=args.device_id,
        )
        transcript_output.parent.mkdir(parents=True, exist_ok=True)
        transcript_output.write_text(transcript + "\n", encoding="utf-8")
        print(f"转写完成: {transcript_output}")
        print(f"转写内容: {transcript}")

        print("步骤 3/4: DeepSeek 总结实验易错点")
        summary = summarize_with_deepseek(
            transcript=transcript,
            model=args.model,
            base_url=args.base_url,
            max_chars=args.summary_max_chars,
        )
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(summary + "\n", encoding="utf-8")
        print(f"总结完成: {summary_output}")
        print(f"播报文案: {summary}")

        print("步骤 4/4: Bert-VITS2 合成并播放")
        run_tts(summary, tts_output)
        if args.skip_playback:
            print(f"已生成音频，未播放: {tts_output}")
        else:
            play_audio(tts_output, args.speaker_device, args.speaker_volume)
            print("播放完成。")

    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
