# VoxCPM Mac 启动指南

## 首次安装（已完成）

```bash
cd /Users/wang2gou/w2gdir/repo/utils/VoxCPM
uv venv voxcpm-env --python 3.10
source voxcpm-env/bin/activate
uv pip install -e .
uv pip install torchcodec
```

## 日常启动

```bash
./start.sh
```

访问：**http://localhost:7860**

## 命令行使用（可选）

```bash
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg/7.1.1_3/lib voxcpm --text "你好世界" --output out.wav
```

## API 服务

### 首次安装 API 依赖

```bash
source voxcpm-env/bin/activate
uv pip install -r requirements-api.txt
```

### 启动 API 服务

```bash
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg/7.1.1_3/lib python api_server.py
```

访问：**http://localhost:8000/docs**（Swagger 文档）

### API 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 文本转语音
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界"}' \
  --output output.wav

# 声音克隆
curl -X POST http://localhost:8000/tts/clone \
  -F "text=你好世界" \
  -F "prompt_text=参考文本内容" \
  -F "prompt_audio=@reference.wav" \
  --output output.wav
```

## 停止服务

```bash
# Ctrl+C 或
lsof -ti:7860 | xargs kill -9
```

## 完全卸载

```bash
rm -rf voxcpm-env models
```

## 注意事项

- 需要 FFmpeg（已通过 Homebrew 安装）
- 使用 Apple Metal (MPS) 加速，速度约 15 it/s
- 首次运行会自动下载模型（约2-3GB）
