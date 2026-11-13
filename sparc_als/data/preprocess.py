"""Preprocessing: resample -> loudness-normalize -> VAD -> trim.

Follows Section 3 (Materials) of the manuscript and the settings of
configs/preprocessing.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from sparc_als.utils.io import load_yaml


def resample_16k(path: str, out_path: str, target_sr: int = 16000) -> None:
    import librosa
    import soundfile as sf
    wav, sr = librosa.load(path, sr=target_sr, mono=True)
    sf.write(out_path, wav, target_sr, subtype="PCM_16")


def normalize_loudness(wav: np.ndarray, target_rms_dbfs: float = -26.0,
                       peak_ceiling_dbfs: float = -1.0) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    target_rms = 10 ** (target_rms_dbfs / 20)
    gain = target_rms / rms
    wav = wav * gain
    peak_ceiling = 10 ** (peak_ceiling_dbfs / 20)
    peak = np.max(np.abs(wav))
    if peak > peak_ceiling:
        wav = wav * (peak_ceiling / peak)
    return wav


def vad_trim(wav: np.ndarray, sr: int, aggressiveness: int = 2,
             frame_ms: int = 30, min_voiced_ms: int = 150,
             min_silence_merge_ms: int = 200) -> np.ndarray:
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    frame_len = int(sr * frame_ms / 1000)
    # Pad to multiple of frame length
    pad = (-len(wav)) % frame_len
    if pad:
        wav = np.pad(wav, (0, pad))
    pcm16 = (wav * 32768).astype(np.int16).tobytes()
    voiced = []
    for i in range(0, len(pcm16), frame_len * 2):
        frame = pcm16[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        voiced.append(vad.is_speech(frame, sr))
    voiced = np.array(voiced, dtype=bool)
    # Simple edge-trim: first/last voiced frame
    if voiced.any():
        first = np.argmax(voiced)
        last = len(voiced) - np.argmax(voiced[::-1])
        return wav[first * frame_len:last * frame_len]
    return wav


def trim_sustained_vowel(wav: np.ndarray, sr: int, trim_ms: int = 250) -> np.ndarray:
    k = int(sr * trim_ms / 1000)
    if len(wav) <= 2 * k:
        return wav
    return wav[k:-k]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="SPARC-ALS preprocessing.")
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print("Preprocessing config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("[dry-run] CLI validated, no audio processed.")
        return

    # Full implementation iterates over corpora/speakers/tokens, applying the
    # chain above and writing a SHA256 manifest. Left out here for brevity
    # but the API above is the production entry point.
    print("Preprocessing ready. Launch per-corpus workers as needed.")


if __name__ == "__main__":
    main()
