#!/usr/bin/env python3
"""
longform_ab.py — Long-form multi-chunk synthesis with locked params + voice anchor,
plus an A/B harness to compare timbre/emotion drift across voice_anchor_strength values.

Fork tool for VoxCPM2. Talks to the REST API (api_server.py) and:
  - registers ONE reference (with transcript -> ref_continuation, the strongest anchor);
  - synthesizes a multi-chunk story via /tts/clone_ref with IDENTICAL params on every
    chunk (point A: locked params + single reference + retry_badcase);
  - sweeps ``voice_anchor_strength`` (point B; 0.0 == upstream behaviour);
  - prints a rough acoustic drift proxy (long-term MFCC similarity to the reference and
    per-chunk pitch/energy) so you can pick a strength.

The metric is only a proxy — the concatenated wavs are the real judge. Listen to them.

Usage:
  python longform_ab.py                                 # built-in story, sweep {0, 0.15, 0.25}
  python longform_ab.py --strengths 0 0.1 0.15 0.2 0.25
  python longform_ab.py --base http://localhost:8000 --out /tmp/voxcpm_ab
  python longform_ab.py --ref my_ref.wav --ref-text "what is said in my_ref.wav"
"""
import argparse
import io
import os
import sys

import numpy as np
import requests
import soundfile as sf

# A reference line synthesized with the default voice when --ref is not given.
DEFAULT_REF_TEXT = (
    "In a small town by the sea, there lived a curious young girl "
    "who loved to watch the ships sail by."
)

# A short story split into chunks with deliberately varied mood, to surface drift.
DEFAULT_STORY = [
    "The morning sun rose gently over the harbor, painting the sky in soft shades of gold and pink.",
    "But as the day wore on, dark clouds gathered, and a cold wind began to howl across the restless water.",
    "By nightfall the storm had passed, and a deep calm returned, leaving the village quiet and at peace once more.",
]

# Locked generation params — point A: every chunk uses EXACTLY these.
LOCKED = dict(cfg_value=2.0, inference_timesteps=5, normalize=False, retry_badcase=True)


def synth_reference(base, text, out_path):
    r = requests.post(f"{base}/tts", json={"text": text, **LOCKED}, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def register_reference(base, wav_path, transcript):
    with open(wav_path, "rb") as f:
        data = {}
        if transcript:
            data["transcript"] = transcript
        r = requests.post(f"{base}/references", files={"audio": f}, data=data, timeout=300)
    r.raise_for_status()
    j = r.json()
    return j["reference_id"], j.get("mode")


def clone_ref(base, ref_id, text, strength):
    payload = {"reference_id": ref_id, "text": text, "voice_anchor_strength": strength, **LOCKED}
    r = requests.post(f"{base}/tts/clone_ref", json=payload, timeout=300)
    r.raise_for_status()
    return r.content, float(r.headers.get("X-Duration-Seconds", 0.0))


def load_wav(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        y, sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32")
    else:
        y, sr = sf.read(path_or_bytes, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def features(y, sr):
    """Long-term acoustic descriptors used as a rough timbre/prosody proxy."""
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)      # (20, T)
    mfcc_mean = mfcc.mean(axis=1)                            # long-term avg -> timbre proxy
    rms = float(librosa.feature.rms(y=y).mean())
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_med = float(np.median(f0)) if f0.size else float("nan")
    except Exception:
        f0_med = float("nan")
    return {"mfcc_mean": mfcc_mean, "rms": rms, "f0": f0_med}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 long-form drift A/B (voice anchor)")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default="/tmp/voxcpm_ab")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.15, 0.25])
    ap.add_argument("--ref", default=None, help="reference wav (else synthesized with default voice)")
    ap.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="transcript of --ref (enables ref_continuation)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) reference
    if args.ref:
        ref_path, ref_text = args.ref, args.ref_text
        print(f"[ref] using {ref_path}")
    else:
        ref_path = os.path.join(args.out, "reference.wav")
        print(f"[ref] synthesizing default-voice reference -> {ref_path}")
        synth_reference(args.base, args.ref_text, ref_path)
        ref_text = args.ref_text
    ref_id, mode = register_reference(args.base, ref_path, ref_text)
    print(f"[ref] reference_id={ref_id} mode={mode}\n")

    ref_y, sr = load_wav(ref_path)
    ref_feat = features(ref_y, sr)

    # 2) sweep strengths; each run uses locked params, only the anchor changes
    summary = []
    for s in args.strengths:
        print(f"=== voice_anchor_strength = {s} ===")
        chunk_feats, chunk_audio = [], []
        for i, text in enumerate(DEFAULT_STORY):
            wav_bytes, dur = clone_ref(args.base, ref_id, text, s)
            p = os.path.join(args.out, f"s{s}_chunk{i}.wav")
            with open(p, "wb") as f:
                f.write(wav_bytes)
            y, _ = load_wav(wav_bytes)
            ft = features(y, sr)
            sim = cos(ft["mfcc_mean"], ref_feat["mfcc_mean"])
            chunk_feats.append((sim, ft["f0"], ft["rms"]))
            chunk_audio.append(y)
            print(f"  chunk{i}: {dur:5.2f}s  sim_to_ref={sim:.4f}  f0={ft['f0']:6.1f}Hz  rms={ft['rms']:.4f}")

        full = np.concatenate(chunk_audio)
        full_path = os.path.join(args.out, f"s{s}_FULL.wav")
        sf.write(full_path, full, sr)

        sims = np.array([c[0] for c in chunk_feats])
        f0s = np.array([c[1] for c in chunk_feats])
        rmss = np.array([c[2] for c in chunk_feats])
        row = dict(
            strength=s,
            sim_mean=float(np.nanmean(sims)),
            sim_std=float(np.nanstd(sims)),          # cross-chunk timbre drift (lower=better)
            f0_std=float(np.nanstd(f0s)),            # pitch drift across chunks
            rms_std=float(np.nanstd(rmss)),          # energy drift across chunks
            full=full_path,
        )
        summary.append(row)
        print(f"  -> {full_path}\n")

    # 3) report
    print("================ DRIFT SUMMARY (lower *_std = less drift) ================")
    print(f"{'strength':>8} | {'sim_to_ref(mean)':>16} | {'sim_std':>8} | {'f0_std(Hz)':>10} | {'rms_std':>8}")
    print("-" * 70)
    for r in summary:
        print(f"{r['strength']:>8} | {r['sim_mean']:>16.4f} | {r['sim_std']:>8.4f} | {r['f0_std']:>10.2f} | {r['rms_std']:>8.4f}")
    print("\nConcatenated wavs to listen to:")
    for r in summary:
        print(f"  strength {r['strength']}: {r['full']}")
    print("\nNote: sim/f0/rms are a coarse proxy. Trust your ears on the *_FULL.wav files.")


if __name__ == "__main__":
    sys.exit(main())
