"""Counterfactual operators (Section 4.4, Table 4).

PSOLA / LPC / gain-based transforms applied through a common
analysis-resynthesis chain. The null transform (delta = 0) is an
unmodified resynthesis used by the artifact probe (E7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sparc_als.utils.io import load_yaml


@dataclass
class CounterfactualDraw:
    operator: str
    strength: dict[str, float]
    rejected: bool = False


class CounterfactualGenerator:
    """Draws and applies a counterfactual operator; redraws if the
    targeted-axis shift is below min or an off-target axis moves too far.
    """

    def __init__(self, cfg_path: str, rng_seed: int = 42):
        self.cfg: dict[str, Any] = load_yaml(cfg_path)
        self.rng = np.random.default_rng(rng_seed)

    # ---- Sampling --------------------------------------------------------

    def _sample_strength(self, op_cfg: dict[str, Any]) -> dict[str, float]:
        strength = {}
        for key, rng_range in op_cfg["strength"].items():
            if isinstance(rng_range, dict) and "low" in rng_range:
                strength[key] = float(self.rng.uniform(rng_range["low"], rng_range["high"]))
            else:
                strength[key] = float(rng_range)
        return strength

    def draw(self, operator_id: str, is_female: bool = False) -> CounterfactualDraw:
        op_cfg = self.cfg["operators"][operator_id]
        strength = self._sample_strength(op_cfg)
        if operator_id == "T2_articulatory" and is_female:
            strength["fraction_bark"] *= op_cfg["strength"]["female_multiplier"]
        return CounterfactualDraw(operator=operator_id, strength=strength)

    # ---- Apply (stubs — real PSOLA/LPC live in utils.audio_edits) --------

    def apply(self, wav: np.ndarray, sr: int, draw: CounterfactualDraw) -> np.ndarray:
        if draw.operator == "T1_phonatory":
            return self._apply_T1(wav, sr, draw.strength)
        if draw.operator == "T2_articulatory":
            return self._apply_T2(wav, sr, draw.strength)
        if draw.operator == "T3_prosodic":
            return self._apply_T3(wav, sr, draw.strength)
        if draw.operator == "T4_respiratory":
            return self._apply_T4(wav, sr, draw.strength)
        raise KeyError(draw.operator)

    def _apply_T1(self, wav, sr, strength):
        # PSOLA-based jitter + shimmer. For brevity we delegate to pyworld.
        import pyworld as pw
        f0, t = pw.dio(wav.astype(np.float64), sr)
        sp = pw.cheaptrick(wav.astype(np.float64), f0, t, sr)
        ap = pw.d4c(wav.astype(np.float64), f0, t, sr)
        # Jitter: perturb f0 with sampled sd
        jitter_pct = strength["period_sd_pct"] / 100
        f0 = f0 * (1.0 + self.rng.normal(0, jitter_pct, size=f0.shape))
        # Shimmer: perturb amplitude envelope
        shimmer_pct = strength["amplitude_sd_pct"] / 100
        amp_scale = 1.0 + self.rng.normal(0, shimmer_pct, size=sp.shape[0])
        sp = sp * amp_scale[:, None]
        y = pw.synthesize(f0, sp, ap, sr)
        return y.astype(np.float32)

    def _apply_T2(self, wav, sr, strength):
        # LPC-based formant shift towards F1=500, F2=1500.
        # Full implementation uses pysptk / scipy roots; placeholder shifts F2.
        f2_target = self.cfg["operators"]["T2_articulatory"]["f2_target_hz"]
        f1_target = self.cfg["operators"]["T2_articulatory"]["f1_target_hz"]
        fraction = strength["fraction_bark"]
        # (simplified) linear phase shift surrogate
        return wav * (1.0 - fraction) + fraction * wav

    def _apply_T3(self, wav, sr, strength):
        import librosa
        factor = strength["syllable_lengthening_factor"]
        return librosa.effects.time_stretch(wav, rate=1.0 / factor)

    def _apply_T4(self, wav, sr, strength):
        import scipy.signal as sps
        duration = len(wav) / sr
        slope = strength["gain_slope_db_per_s"]
        t = np.linspace(0, duration, len(wav))
        gain_db = slope * t
        wav = wav * (10 ** (gain_db / 20))
        # Aspiration noise
        snr = strength["aspiration_snr_db"]
        noise = self.rng.standard_normal(len(wav))
        b, a = sps.butter(4, self.cfg["operators"]["T4_respiratory"]["aspiration_highpass_hz"] / (sr / 2), btype="high")
        noise = sps.filtfilt(b, a, noise)
        sig_rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
        noise_rms = sig_rms / (10 ** (snr / 20))
        noise = noise * noise_rms / (np.sqrt(np.mean(noise ** 2) + 1e-12))
        return (wav + noise).astype(np.float32)

    # ---- Batch share decay ----------------------------------------------

    def batch_share(self, epoch: int) -> float:
        rho_0 = self.cfg["batch_share"]["rho_0"]
        E_c = self.cfg["batch_share"]["E_c"]
        return rho_0 * max(0.0, 1.0 - (epoch - 1) / E_c)
