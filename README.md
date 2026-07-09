# 光明哨兵

基于瑞芯微 RK3588 的化工实验室辅助操作及行为监控系统。项目面向实验室安全场景，结合视频检测、语音识别、内容总结和语音播报，实现违规行为监控与实验操作辅助提醒。

## 功能简介

- 实时视频流采集与 Web 展示
- 基于 YOLOv8/RKNN 的实验室违规行为检测
- 违规事件截图、记录与同步
- 麦克风录音、Whisper 语音转写
- DeepSeek 总结实验操作提醒
- Bert-VITS2/RKNN 语音播报

## 目录说明

- `flask_app.py`：视频检测与 Web 展示主程序
- `harmony_api_server.py`：事件与接口服务
- `harmony_stream_server.py`：视频流服务
- `chem_lab_voice_pipeline.py`：录音、转写、总结、播报一体化流程
- `record_and_transcribe_whisper.py`：录音与 Whisper 转写
- `func/`：YOLOv8 推理相关代码
- `rknnpool/`：RKNN 模型池封装
- `whisper/`：RKNN Whisper 调用代码
- `Bert-VITS2-RKNN2/`：语音合成与播放相关代码

## 运行环境

- 瑞芯微 RK3588 开发板
- Python 3
- RKNN Runtime / RKNN Toolkit Lite2
- OpenCV、Flask、OpenAI SDK 等 Python 依赖
- 摄像头、麦克风和扬声器等外设

模型文件、视频、音频和运行结果未纳入仓库，需要按实际部署环境放置到对应目录。

## 基本使用

```bash
# 启动视频检测与 Web 服务
python flask_app.py

# 运行语音辅助流程
export DEEPSEEK_API_KEY="your_api_key"
python chem_lab_voice_pipeline.py
```

具体设备编号、模型路径和远端服务地址可在脚本顶部配置项或环境变量中调整。
