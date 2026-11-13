"""Acoustic proxy feature extractors — Praat/Parselmouth bindings.

Implements the components of Table 3 of the manuscript. Each function
returns a scalar component for one recording; aggregation into axis
targets t_j (Eq. 5) is handled by `ndt_targets.build_targets`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
    _HAS_PRAAT = True
except ImportError:  # pragma: no cover
    _HAS_PRAAT = False


@dataclass
class PraatPitchSettings:
    method: str = "cc"
    time_step: float = 0.010
    f0_min_male: float = 75.0
    f0_max_male: float = 300.0
    f0_min_female: float = 100.0
    f0_max_female: float = 500.0


def _make_pitch(snd, settings: PraatPitchSettings, sex: str):
    fmin = settings.f0_min_male if sex.startswith("m") else settings.f0_min_female
    fmax = settings.f0_max_male if sex.startswith("m") else settings.f0_max_female
    if settings.method == "cc":
        return snd.to_pitch_cc(time_step=settings.time_step,
                               pitch_floor=fmin, pitch_ceiling=fmax)
    return snd.to_pitch_ac(time_step=settings.time_step,
                           pitch_floor=fmin, pitch_ceiling=fmax)


# --- Axis (i) Phonatory stability -----------------------------------------

def local_jitter(wav: np.ndarray, sr: int, sex: str) -> float:
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    pitch = _make_pitch(snd, PraatPitchSettings(), sex)
    pulses = call([snd, pitch], "To PointProcess (cc)")
    return call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.020, 1.3)


def local_shimmer(wav: np.ndarray, sr: int, sex: str) -> float:
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    pitch = _make_pitch(snd, PraatPitchSettings(), sex)
    pulses = call([snd, pitch], "To PointProcess (cc)")
    return call([snd, pulses], "Get shimmer (local)",
                0.0, 0.0, 0.0001, 0.020, 1.3, 1.6)


def hnr(wav: np.ndarray, sr: int) -> float:
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    harm = snd.to_harmonicity_cc(time_step=0.010, minimum_pitch=75.0,
                                 silence_threshold=0.1, periods_per_window=1.0)
    arr = harm.values[harm.values != -200]  # Praat flag
    return float(np.mean(arr)) if arr.size else 0.0


def f0_sd(wav: np.ndarray, sr: int, sex: str) -> float:
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    pitch = _make_pitch(snd, PraatPitchSettings(), sex)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.std(f0)) if f0.size else 0.0


# --- Axis (ii) Articulatory precision -------------------------------------

def centralization_ratio(formants_per_vowel: dict[str, tuple[float, float]]) -> float:
    """Formant Centralization Ratio (Sapir et al. 2010).

    Requires {a, e, i, u}. Returns NaN otherwise.
    """
    needed = {"a", "e", "i", "u"}
    if not needed.issubset(formants_per_vowel):
        return float("nan")
    F2_i, F2_u = formants_per_vowel["i"][1], formants_per_vowel["u"][1]
    F1_i, F1_u = formants_per_vowel["i"][0], formants_per_vowel["u"][0]
    F2_a, F2_e = formants_per_vowel["a"][1], formants_per_vowel["e"][1]
    F1_a, F1_e = formants_per_vowel["a"][0], formants_per_vowel["e"][0]
    num = F2_u + F2_a + F1_i + F1_u
    den = F2_i + F1_a
    return num / den if den > 1e-6 else float("nan")


def ai_bark_distance(f_a: tuple[float, float], f_i: tuple[float, float]) -> float:
    """Euclidean distance between /a/ and /i/ in Bark (F1, F2)."""
    def hz2bark(f: float) -> float:
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
    a_bark = np.array([hz2bark(f_a[0]), hz2bark(f_a[1])])
    i_bark = np.array([hz2bark(f_i[0]), hz2bark(f_i[1])])
    return float(np.linalg.norm(a_bark - i_bark))


# --- Axis (iii) Prosodic / timing -----------------------------------------

def ddk_rate(wav: np.ndarray, sr: int) -> float:
    """Syllables per second over a /pa-ta-ka/ recording."""
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    intensity = snd.to_intensity(minimum_pitch=100.0, time_step=0.005)
    values = intensity.values[0]
    # Peak detection: 2 dB above neighbouring dips
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            # local dips
            left_dip = values[max(0, i - 10):i].min()
            right_dip = values[i + 1:min(len(values), i + 11)].min()
            if values[i] - max(left_dip, right_dip) > 2.0:
                peaks.append(i * intensity.time_step)
    duration = len(wav) / sr
    return len(peaks) / duration if duration > 0 else 0.0


def interval_variation(wav: np.ndarray, sr: int) -> float:
    """Coefficient of variation of inter-peak intervals."""
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    intensity = snd.to_intensity(minimum_pitch=100.0, time_step=0.005)
    values = intensity.values[0]
    peak_times = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            left_dip = values[max(0, i - 10):i].min()
            right_dip = values[i + 1:min(len(values), i + 11)].min()
            if values[i] - max(left_dip, right_dip) > 2.0:
                peak_times.append(i * intensity.time_step)
    if len(peak_times) < 3:
        return float("nan")
    intervals = np.diff(peak_times)
    return float(np.std(intervals) / np.mean(intervals)) if intervals.mean() > 0 else float("nan")


# --- Axis (iv) Respiratory / phonatory support ----------------------------

def intensity_slope(wav: np.ndarray, sr: int) -> float:
    """Linear-fit dB/s slope of intensity over a sustained /a/."""
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    intensity = snd.to_intensity(minimum_pitch=100.0, time_step=0.010)
    values = intensity.values[0]
    t = np.arange(len(values)) * intensity.time_step
    if len(values) < 3:
        return float("nan")
    slope = np.polyfit(t, values, 1)[0]
    return float(slope)


def cpps(wav: np.ndarray, sr: int) -> float:
    """Smoothed Cepstral Peak Prominence, 60-330 Hz peak search."""
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth is required for acoustic features")
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    power_cepstrogram = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    return call(power_cepstrogram, "Get CPPS", True, 0.01, 0.001,
                60, 330, 0.05, "Parabolic", 0.001, 0, "Exponential decay",
                "Robust")
