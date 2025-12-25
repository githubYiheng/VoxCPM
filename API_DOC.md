# VoxCPM API 文档

## 概述

VoxCPM API 提供文本转语音（TTS）和声音克隆服务。

- **基础地址**: `http://localhost:8000`
- **文档地址**: `http://localhost:8000/docs`（Swagger UI）

---

## 快速开始

### 启动服务

```bash
cd /path/to/VoxCPM
source voxcpm-env/bin/activate
DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ffmpeg/7.1.1_3/lib python api_server.py
```

### 最简单的调用

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是一段测试语音。"}' \
  --output output.wav
```

---

## API 端点

### 1. 健康检查

检查服务和模型状态。

**请求**
```
GET /health
```

**响应**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "message": "Service is running"
}
```

**状态说明**
| status | 说明 |
|--------|------|
| `healthy` | 服务正常，模型已加载 |
| `loading` | 模型正在加载中 |
| `unhealthy` | 服务异常 |

---

### 2. 模型信息

获取当前模型的详细信息。

**请求**
```
GET /info
```

**响应**
```json
{
  "model_id": "openbmb/VoxCPM1.5",
  "sample_rate": 44100,
  "device": "mps",
  "dtype": "bfloat16",
  "lora_enabled": false,
  "denoiser_available": true
}
```

---

### 3. 文本转语音（TTS）

将文本转换为语音，使用模型默认音色。

**请求**
```
POST /tts
Content-Type: application/json
```

**请求体**
```json
{
  "text": "要合成的文本内容",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "output_format": "wav"
}
```

**参数说明**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本（1-10000字符） |
| `cfg_value` | float | ❌ | 2.0 | CFG 引导值（1.0-5.0），越高越稳定 |
| `inference_timesteps` | int | ❌ | 10 | 推理步数（4-50），越高质量越好但更慢 |
| `min_len` | int | ❌ | 2 | 最小输出长度 |
| `max_len` | int | ❌ | 4096 | 最大输出长度 |
| `normalize` | bool | ❌ | false | 是否进行文本正则化 |
| `retry_badcase` | bool | ❌ | true | 生成异常时是否重试 |
| `retry_badcase_max_times` | int | ❌ | 3 | 最大重试次数 |
| `retry_badcase_ratio_threshold` | float | ❌ | 6.0 | 异常检测阈值 |
| `output_format` | string | ❌ | "wav" | 输出格式：wav / mp3 / flac |

**响应**

返回音频文件（二进制），响应头包含元信息：

| 响应头 | 说明 |
|--------|------|
| `X-Sample-Rate` | 采样率（如 44100） |
| `X-Duration-Seconds` | 音频时长（秒） |
| `X-Text-Length` | 输入文本长度 |
| `Content-Type` | audio/wav 或 audio/mpeg 或 audio/flac |

---

### 4. 声音克隆

使用参考音频克隆特定声音。

**请求**
```
POST /tts/clone
Content-Type: multipart/form-data
```

**参数说明**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt_audio` | file | ✅ | - | 参考音频文件（wav/mp3/flac/ogg/m4a） |
| `prompt_text` | string | ✅ | - | 参考音频中说的文字内容 |
| `text` | string | ✅ | - | 要合成的目标文本 |
| `denoise` | bool | ❌ | false | 是否对参考音频降噪 |
| `cfg_value` | float | ❌ | 2.0 | CFG 引导值 |
| `inference_timesteps` | int | ❌ | 10 | 推理步数 |
| `normalize` | bool | ❌ | false | 文本正则化 |
| `output_format` | string | ❌ | "wav" | 输出格式 |

**响应**

返回音频文件（二进制）。

---

## 使用示例

### 场景 1：基础文本转语音

最简单的用法，使用默认音色和参数。

**curl**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "欢迎使用 VoxCPM 语音合成服务。"}' \
  --output basic.wav
```

**Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "欢迎使用 VoxCPM 语音合成服务。"}
)

with open("basic.wav", "wb") as f:
    f.write(response.content)
```

---

### 场景 2：调整生成参数

提高质量（更多推理步数）或调整风格（CFG 值）。

**curl**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一段高质量的语音合成示例。",
    "cfg_value": 2.5,
    "inference_timesteps": 20,
    "output_format": "mp3"
  }' \
  --output high_quality.mp3
```

**Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "这是一段高质量的语音合成示例。",
        "cfg_value": 2.5,
        "inference_timesteps": 20,
        "output_format": "mp3"
    }
)

with open("high_quality.mp3", "wb") as f:
    f.write(response.content)
```

---

### 场景 3：声音克隆（基础）

使用一段参考音频，让模型模仿该声音。

**准备工作**
1. 准备一段清晰的参考音频（3-10秒为佳）
2. 准确写出参考音频中说的文字内容

**curl**
```bash
curl -X POST http://localhost:8000/tts/clone \
  -F "prompt_audio=@/path/to/reference.wav" \
  -F "prompt_text=这是参考音频中说的原文" \
  -F "text=这是我想让克隆的声音说的新内容" \
  --output cloned.wav
```

**Python**
```python
import requests

with open("reference.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/tts/clone",
        files={"prompt_audio": audio_file},
        data={
            "prompt_text": "这是参考音频中说的原文",
            "text": "这是我想让克隆的声音说的新内容"
        }
    )

with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

---

### 场景 4：声音克隆 + 降噪

参考音频有噪音时，启用降噪功能。

**curl**
```bash
curl -X POST http://localhost:8000/tts/clone \
  -F "prompt_audio=@noisy_reference.wav" \
  -F "prompt_text=参考音频的文字内容" \
  -F "text=要合成的新内容" \
  -F "denoise=true" \
  --output cloned_denoised.wav
```

**Python**
```python
import requests

with open("noisy_reference.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/tts/clone",
        files={"prompt_audio": audio_file},
        data={
            "prompt_text": "参考音频的文字内容",
            "text": "要合成的新内容",
            "denoise": "true"
        }
    )

with open("cloned_denoised.wav", "wb") as f:
    f.write(response.content)
```

---

### 场景 5：声音克隆 + 完整参数

使用所有可调参数进行精细控制。

**curl**
```bash
curl -X POST http://localhost:8000/tts/clone \
  -F "prompt_audio=@reference.wav" \
  -F "prompt_text=大家好，欢迎收听今天的节目" \
  -F "text=今天我们要讨论的话题是人工智能的未来发展" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=15" \
  -F "denoise=true" \
  -F "normalize=false" \
  -F "output_format=flac" \
  --output podcast.flac
```

**Python**
```python
import requests

with open("reference.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/tts/clone",
        files={"prompt_audio": ("reference.wav", audio_file, "audio/wav")},
        data={
            "prompt_text": "大家好，欢迎收听今天的节目",
            "text": "今天我们要讨论的话题是人工智能的未来发展",
            "cfg_value": 2.0,
            "inference_timesteps": 15,
            "denoise": "true",
            "normalize": "false",
            "output_format": "flac"
        }
    )

with open("podcast.flac", "wb") as f:
    f.write(response.content)
```

---

### 场景 6：批量生成

循环调用 API 批量生成多段语音。

**Python**
```python
import requests
from pathlib import Path

texts = [
    "第一段要合成的内容。",
    "第二段要合成的内容。",
    "第三段要合成的内容。",
]

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for i, text in enumerate(texts):
    response = requests.post(
        "http://localhost:8000/tts",
        json={"text": text}
    )

    if response.status_code == 200:
        output_path = output_dir / f"output_{i+1}.wav"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Generated: {output_path}")
    else:
        print(f"Failed for text {i+1}: {response.text}")
```

---

### 场景 7：异步调用（Python asyncio）

高效的异步批量调用。

**Python**
```python
import asyncio
import aiohttp

async def generate_tts(session, text, output_path):
    async with session.post(
        "http://localhost:8000/tts",
        json={"text": text}
    ) as response:
        if response.status == 200:
            content = await response.read()
            with open(output_path, "wb") as f:
                f.write(content)
            return True
    return False

async def main():
    texts = [
        ("你好世界", "hello.wav"),
        ("今天天气真好", "weather.wav"),
        ("谢谢大家", "thanks.wav"),
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [
            generate_tts(session, text, path)
            for text, path in texts
        ]
        results = await asyncio.gather(*tasks)
        print(f"Completed: {sum(results)}/{len(results)}")

asyncio.run(main())
```

---

## 错误处理

### 错误响应格式

```json
{
  "error": "错误类型",
  "detail": "详细错误信息",
  "error_code": "ERROR_CODE"
}
```

### 常见错误

| HTTP 状态码 | error_code | 说明 | 解决方案 |
|-------------|------------|------|----------|
| 400 | VALIDATION_ERROR | 参数验证失败 | 检查请求参数格式 |
| 400 | FILE_TOO_LARGE | 文件过大 | 上传小于 50MB 的文件 |
| 400 | UNSUPPORTED_FORMAT | 不支持的音频格式 | 使用 wav/mp3/flac/ogg/m4a |
| 500 | GENERATION_FAILED | 生成失败 | 检查文本内容，重试 |
| 503 | MODEL_NOT_LOADED | 模型未加载 | 等待模型加载完成 |
| 504 | TIMEOUT | 请求超时 | 减少文本长度或增加超时 |

### 错误处理示例

**Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "测试文本"}
)

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Success!")
else:
    error = response.json()
    print(f"Error: {error['error']}")
    print(f"Detail: {error.get('detail', 'N/A')}")
```

---

## 参考音频最佳实践

### 音频要求

| 项目 | 建议 |
|------|------|
| **时长** | 3-10 秒（太短效果差，太长处理慢） |
| **格式** | WAV（推荐）、MP3、FLAC |
| **采样率** | 16kHz 以上 |
| **音质** | 清晰无噪音（或启用 denoise） |
| **内容** | 自然语速的完整句子 |

### prompt_text 要求

- **必须准确**：与参考音频内容完全一致
- **完整标点**：保持正确的标点符号
- **语言一致**：与目标文本使用相同语言

### 效果优化

1. **选择代表性音频**：选择能体现说话人特征的音频
2. **避免特殊音效**：不要使用有背景音乐的音频
3. **文本对齐**：prompt_text 必须精确匹配音频内容
4. **启用降噪**：如果音频有噪音，设置 `denoise=true`

---

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | 0.0.0.0 | 服务监听地址 |
| `API_PORT` | 8000 | 服务端口 |
| `HF_MODEL_ID` | openbmb/VoxCPM1.5 | HuggingFace 模型 ID |
| `MODEL_PATH` | - | 本地模型路径（覆盖 HF_MODEL_ID） |
| `LOAD_DENOISER` | true | 是否加载降噪模型 |
| `MAX_CONCURRENT_REQUESTS` | 1 | 最大并发请求数 |
| `REQUEST_TIMEOUT_SECONDS` | 300 | 请求超时时间（秒） |
| `MAX_UPLOAD_SIZE_MB` | 50 | 最大上传文件大小（MB） |

---

## 性能参考

在 Apple M4 Max（MPS 加速）上的参考性能：

| 文本长度 | inference_timesteps | 预计耗时 |
|----------|---------------------|----------|
| 20 字 | 10 | ~5 秒 |
| 50 字 | 10 | ~10 秒 |
| 100 字 | 10 | ~20 秒 |
| 100 字 | 20 | ~40 秒 |

> 实际时间因硬件和文本内容而异。
