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
./start_api.sh
```

`start_api.sh` 已经设置好 `DYLD_LIBRARY_PATH`、激活 `voxcpm-env`、并把 `MODEL_PATH` 指向本地 `models/openbmb__VoxCPM2`,加载完成后监听 `http://0.0.0.0:8000`。

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
  "model_id": "models/VoxCPM2",
  "sample_rate": 48000,
  "device": "mps",
  "dtype": "float32",
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
  "inference_timesteps": 5,
  "output_format": "wav"
}
```

**参数说明**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的文本（1-10000字符） |
| `cfg_value` | float | ❌ | 2.0 | CFG 引导值（1.0-5.0），越高越稳定 |
| `inference_timesteps` | int | ❌ | 5 | 推理步数（4-50），越高质量越好但更慢。MPS 上 5 是速度/质量甜点 |
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
| `X-Sample-Rate` | 采样率（如 48000） |
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
| `inference_timesteps` | int | ❌ | 5 | 推理步数 |
| `normalize` | bool | ❌ | false | 文本正则化 |
| `output_format` | string | ❌ | "wav" | 输出格式 |

**响应**

返回音频文件（二进制）。

> ⚠️ **每次调用都会重新对参考音频做 VAE 编码**,跨多个 chunk 会浪费。需要复用同一段 reference 时,改用 [5. References 缓存](#5-references-缓存)+ [6. /tts/clone_ref](#6-ttsclone_ref-基于-reference_id-合成) 或 [7. /tts/chain](#7-ttschain-链式合成零音色漂移)。

---

### 5. References 缓存

注册一段参考音频,服务端只做一次 VAE 编码,之后用 `reference_id` 反复调用 `/tts/clone_ref` 和 `/tts/chain`。

#### 5.1 注册 reference

**请求**
```
POST /references
Content-Type: multipart/form-data
```

**参数说明**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio` | file | ✅ | 参考音频(wav/mp3/flac/ogg/m4a) |
| `transcript` | string | ❌ | 参考音频对应的转写文本(1-2000 字符)。提供时启用 **Ultimate Cloning 三通道**(prompt_wav + reference_wav + prompt_text),不提供时为 reference-only 模式 |

**响应(JSON)**
```json
{
  "reference_id": "ref_c6c7364fba35",
  "mode": "ref_continuation",
  "transcript": "Hello, this is a VoxCPM 2 smoke test.",
  "audio_filename": "voxcpm2_smoke.wav",
  "created_at": 1779187608.011
}
```

`mode` 字段:
- `ref_continuation`:有 transcript,三通道模式,**官方推荐用于最大克隆相似度**
- `reference`:无 transcript,仅 reference_wav_path

**约束**:服务端 LRU 上限 32 个 reference,超出按最近最少使用淘汰;重启服务会全部丢失。

---

#### 5.2 列出 references

**请求**
```
GET /references
```

返回最近最先的列表。

---

#### 5.3 删除 reference

**请求**
```
DELETE /references/{reference_id}
```

返回 `{"deleted": "ref_xxx"}` 或 404。

---

### 6. /tts/clone_ref — 基于 reference_id 合成

复用已注册的 reference,**跳过 VAE 编码,生成更快**。

**请求**
```
POST /tts/clone_ref
Content-Type: application/json
```

**请求体**
```json
{
  "reference_id": "ref_c6c7364fba35",
  "text": "用这段克隆音色读这句话。",
  "cfg_value": 2.0,
  "inference_timesteps": 5,
  "output_format": "wav"
}
```

**参数说明**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `reference_id` | string | ✅ | - | 来自 `POST /references` |
| `text` | string | ✅ | - | 要合成的文本(1-10000) |
| 其它参数 | — | ❌ | — | 与 `/tts` 相同(`cfg_value` / `inference_timesteps` / `min_len` / `max_len` / `normalize` / `retry_badcase*` / `output_format`) |

**响应**

返回音频二进制,响应头额外含 `X-Reference-Id`。其它响应头与 `/tts` 相同。

**错误**
- `404`:`reference_id` 不存在或已被淘汰
- `503`:模型未加载

---

### 7. /tts/chain — 链式合成(零音色漂移)

把长文本切分为多段时,**用上一段的输出 audio feature(in-memory)作为下一段的 prompt**,而不是每段独立 clone。可以最大程度避免段间音色漂移。底层用 VoxCPM2 的 `merge_prompt_cache` 实现,**音频特征在显存里直接 concat,完全无 wav 编解码 round-trip**。

**请求**
```
POST /tts/chain
Content-Type: application/json
```

**请求体**
```json
{
  "reference_id": "ref_c6c7364fba35",
  "parent_chunk_id": "chunk_45337404ddd5",
  "text": "这是后续的一段话。"
}
```

**参数说明**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `reference_id` | string | ✅ | 锚定的参考音色(始终一致) |
| `parent_chunk_id` | string | ❌ | 上一次 `/tts/chain` 响应的 `X-Chunk-Id`。第一段不传,从 reference 起步;后续段传上次的 chunk_id |
| `text` | string | ✅ | 当前段文本(1-10000) |
| 其它 | — | ❌ | 同 `/tts` |

**响应**

返回音频二进制。响应头:

| 响应头 | 说明 |
|--------|------|
| `X-Chunk-Id` | 本次生成的 chunk id,传给下一段的 `parent_chunk_id` |
| `X-Chain-Started` | `true` 表示链路新起,`false` 表示继承自 `parent_chunk_id` |
| `X-Reference-Id` | 回显 reference_id |
| 其它 | 同 `/tts` |

**约束**:
- chunk 缓存 LRU 上限 200 条
- 单条 **TTL 30 分钟**,过期不能再被引用
- 服务端重启,所有 chunk 链丢失;客户端需要重建

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

### 场景 8:reference 缓存复用(多次调用同一音色)

注册一次,后续 N 次合成都不重新做 VAE 编码,每次省约 1-2 秒。

**Python**
```python
import requests

BASE = "http://localhost:8000"

# 1) 注册 reference(只做一次)
with open("ref.wav", "rb") as f:
    resp = requests.post(
        f"{BASE}/references",
        files={"audio": f},
        data={"transcript": "参考音频的转写文本"},
    )
ref_id = resp.json()["reference_id"]

# 2) 反复用同一 reference 合成多段(无文件上传)
texts = ["第一段。", "第二段。", "第三段。"]
for i, text in enumerate(texts):
    out = requests.post(
        f"{BASE}/tts/clone_ref",
        json={"reference_id": ref_id, "text": text},
    )
    open(f"out_{i:03d}.wav", "wb").write(out.content)

# 3) 用完释放
requests.delete(f"{BASE}/references/{ref_id}")
```

---

### 场景 9:长文本链式合成(零音色漂移)

把长文本切成多段,**每段以上一段的音频特征作为前缀**,模型在 in-memory feature 上继续合成,段间音色高度一致。

**Python**
```python
import requests

BASE = "http://localhost:8000"

# 1) 注册 reference(锚定音色)
with open("ref.wav", "rb") as f:
    ref_id = requests.post(
        f"{BASE}/references",
        files={"audio": f},
        data={"transcript": "参考音频的转写文本"},
    ).json()["reference_id"]

# 2) 长文本切 chunk(按句号/段落,推荐每段 30-60 秒对应的字数)
chunks = [
    "第一段:这是一段较长的旁白,会被作为整个合成的开场。",
    "第二段:接续上文,叙事继续,音色应当与第一段保持一致。",
    "第三段:故事接近尾声,情绪逐步收束,音色仍保持稳定。",
]

# 3) 链式调用:首段不带 parent_chunk_id;后续段传上次的 X-Chunk-Id
parent = None
for i, text in enumerate(chunks):
    payload = {"reference_id": ref_id, "text": text}
    if parent:
        payload["parent_chunk_id"] = parent
    r = requests.post(f"{BASE}/tts/chain", json=payload)
    r.raise_for_status()
    open(f"chunk_{i:03d}.wav", "wb").write(r.content)
    parent = r.headers["X-Chunk-Id"]      # 串到下一段
    print(f"chunk {i}: chunk_id={parent}, duration={r.headers['X-Duration-Seconds']}s")
```

**注意事项**:
- chunk 缓存 TTL 30 分钟,长链路要避免单段处理超过这个时间
- chunk 切分尽量按 **句号/段落** 切,避免在词中或语义未尽处硬截
- 推荐**每段 30-90 秒**对应的文本长度(中文 ~100-300 字 / 英文 ~400-1000 chars)
- 同一链路内,**所有调用必须复用相同的** `cfg_value` / `inference_timesteps` / `normalize`,任一参数变化都会引入可感的音色/能量差异

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
| `HF_MODEL_ID` | openbmb/VoxCPM2 | HuggingFace 模型 ID |
| `MODEL_PATH` | models/VoxCPM2 | 本地模型路径（覆盖 HF_MODEL_ID） |
| `LOAD_DENOISER` | true | 是否加载降噪模型 |
| `MAX_CONCURRENT_REQUESTS` | 1 | 最大并发请求数 |
| `REQUEST_TIMEOUT_SECONDS` | 300 | 请求超时时间（秒） |
| `MAX_UPLOAD_SIZE_MB` | 50 | 最大上传文件大小（MB） |

---

## 性能参考

在 Apple Silicon(MPS 加速)上的实测参考(VoxCPM2, 48kHz):

| 端点 | inference_timesteps | 文本(英) | 音频时长 | 单次耗时 | RTF |
|------|---------------------|---------|---------|---------|-----|
| `/tts`(默认) | 5 | 55 chars | ~4 s | ~12 s | 3× |
| `/tts` | 10 | 55 chars | ~4 s | ~59 s | 15× |
| `/tts/clone`(每次上传) | 5 | 55 chars | ~4 s | ~12-14 s | 3× |
| `/tts/clone_ref`(缓存命中) | 5 | 55 chars | ~4 s | **~10 s** | 2.5× |
| `/tts/chain` step N | 5 | 55 chars | ~4 s | **~9 s** | 2.3× |

**优化要点**:
- 默认 `inference_timesteps=5` 是速度/质量甜点;调到 10 慢 5×;调到 4 与 5 几乎一样
- 同一 reference 复用 `clone_ref` 比 `/tts/clone` 单次省 ~1-2 秒(跳过 VAE encode)
- `chain` 第 N 段不随链长变慢(`merge_prompt_cache` 是 in-memory concat,无重复开销)
- 实际时间因硬件、文本内容、首次 LM 缓存预热而异
