# 多段合成防漂移指南

> 把长文本切成多段分别合成、拼起来后发现**各段音色/情绪有细微差异**(同一个人却像换了状态)？
> 照这份文档调用 API,就能把漂移压到最低。配合 [`API_DOC.md`](API_DOC.md) 看完整参数。

---

## 一句话原因

长文本必须切成多段,而**每段是各自独立生成的**。参考音频只锁住了"**是谁**在说",
但每段的语调、情绪、能量是合成时当场重新生成的——所以段与段之间会飘。

---

## ✅ 推荐做法(绝大多数情况用这个)

**注册一次参考音 → 每段都用 `clone_ref`,全程锁死参数 + 开启 voice anchor。**

三步:

1. **注册参考音(只做一次)** `POST /references`,尽量带上 `transcript`(参考音的原话),克隆相似度最高。
2. **每段都用 `POST /tts/clone_ref`**,且每段都满足下面的[四条铁律](#四条铁律)。
3. 把各段输出按顺序拼接即可。

### 完整示例(可直接跑)

```python
import requests

BASE = "http://localhost:8000"

# 所有段共用这组参数 —— 全程不要改任何一个值
LOCKED = {
    "cfg_value": 2.0,
    "inference_timesteps": 5,
    "normalize": False,
    "voice_anchor_strength": 0.15,   # ← 防漂移开关,0.15 起步
}

# 1) 注册参考音(只做一次),拿到 reference_id
ref_id = requests.post(
    f"{BASE}/references",
    files={"audio": open("reference.wav", "rb")},
    data={"transcript": "What is actually said in reference.wav."},  # 带上最佳
).json()["reference_id"]

# 2) 长文本按句号/段落切好
segments = [
    "The morning sun rose gently over the quiet harbor.",
    "But as the day wore on, dark clouds gathered over the water.",
    "By nightfall the storm had passed, and calm returned at last.",
]

# 3) 每段都复用同一个 reference_id + 同一组 LOCKED 参数
for i, text in enumerate(segments):
    r = requests.post(f"{BASE}/tts/clone_ref",
                      json={"reference_id": ref_id, "text": text, **LOCKED})
    open(f"seg_{i:02d}.wav", "wb").write(r.content)
```

---

## 四条铁律

无论用哪种方式,这四条都要守住,否则照样漂:

| # | 铁律 | 为什么 |
|---|------|--------|
| 1 | **全程复用同一个 `reference_id`** | 每段重新上传/重新注册都会引入差异 |
| 2 | **所有段 `cfg_value` / `inference_timesteps` / `normalize` 完全一致** | 任意一个值变了,音色/能量就会有可感差异 |
| 3 | **`voice_anchor_strength` 设 0.15 起步,按耳朵微调** | 它把音色持续拉回参考音,直接对抗漂移 |
| 4 | **按句号/段落切分**,别在词中间或半句话硬截 | 断点不自然会放大段间差异 |

---

## voice anchor 强度怎么调

`voice_anchor_strength` 就是防漂移的旋钮(0 = 关,等于不做任何处理):

| 取值 | 效果 |
|------|------|
| `0.0` | 关闭(默认),漂移不处理 |
| `0.10` | 轻度锚定;若 0.15 听起来发闷,退到这里 |
| **`0.15`** | **推荐起点**,多数场景够用 |
| `0.25` | 强锚定;0.15 还在漂时加到这里 |

**调法**:先用 `0.15` 整篇跑一遍 → 用耳朵听拼接结果 →
还漂就加到 `0.25`;若声音发闷/沉就退到 `0.10`。**最终以听感为准。**

---

## 进阶:要段间"一气呵成"用 chain

如果是有声书/旁白这类**追求段间极致连贯**的场景,可以用 `POST /tts/chain`:
它让每段承接上一段的输出特征,段间过渡最自然(可同时叠加 `voice_anchor_strength`)。

⚠️ **长故事的坑**:chain 的上下文会随链长**累积增长**,串太长会变慢、最终撑爆模型上限。
所以**别把整篇长文串成一条无限链**——每隔几段(一个场景/自然段)就从 `reference_id` 重起一条新链,
把"重起点"放在场景切换或停顿处(那里即使有极微小变化也听不出来)。详见 [`API_DOC.md`](API_DOC.md) 的 `/tts/chain` 一节。

---

## 选哪个

| 你的情况 | 用 |
|----------|-----|
| 大多数情况、分段批量、要简单稳妥 | **`clone_ref` + `voice_anchor_strength=0.15`** ✅ |
| 有声书/旁白,追求段间极致连贯,且愿意处理"重起链" | **`chain`**(可叠加 anchor) |

---

## 上线前自检清单

- [ ] 参考音只注册了一次,所有段用同一个 `reference_id`
- [ ] 注册时带了准确的 `transcript`
- [ ] 每段的 `cfg_value` / `inference_timesteps` / `normalize` 完全相同
- [ ] 每段都带了 `voice_anchor_strength`(且取值相同)
- [ ] 文本按句号/段落切,没有半句硬截
- [ ] 用耳朵听过拼接结果,强度已调到满意

> 仓库内 `longform_ab.py` 可一键扫不同强度并产出对比音频,方便定值。
