# VoxCPM2 Mac 启动指南

当前版本：**VoxCPM 2.0.3**（模型 VoxCPM2，2B 参数，48kHz）

## 首次安装（已完成）

```bash
cd /Users/wang2gou/w2gdir/repo/utils/VoxCPM
uv venv voxcpm-env --python 3.10
source voxcpm-env/bin/activate
uv pip install -e .
uv pip install torchcodec
```

## 模型下载（已完成，从 ModelScope，国内更快）

```bash
source voxcpm-env/bin/activate
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM2', local_dir='models/VoxCPM2')"
```

模型落在 `models/VoxCPM2`（约 4.3GB）。`start.sh` 和 `api_server.py` 默认都用这个本地路径，不再走 HuggingFace。

## 日常启动

```bash
./start.sh
```

访问：**http://localhost:7860**

> 注：v2 的 `app.py` 默认端口是 8808，`start.sh` 里已用 `--port 7860` 固定回原来的端口。

## 命令行使用（可选）

```bash
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg/7.1.1_3/lib voxcpm --text "你好世界" --output out.wav --model-path models/VoxCPM2
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

- 需要 FFmpeg（已通过 Homebrew 安装）；命令前的 `DYLD_LIBRARY_PATH` 指向 ffmpeg 库，brew 升级 ffmpeg 后版本号 `7.1.1_3` 需同步更新
- 使用 Apple Metal (MPS) 加速；VoxCPM2 在 MPS 上自动用 float32（bf16 有数值漂移）
- **实测 RTF ≈ 0.80（M4 Max）**，约比实时快 1.25 倍；扩散步速 ~8 it/s
- 首次运行已下载模型（VoxCPM2 ~4.3GB，辅助 ASR/降噪模型在 `~/.cache/modelscope`）
