import math

import numpy as np
from scipy.signal import periodogram
from torch import nn
from torch.nn import functional as F

HORIZONS = (16, 24, 32, 48, 64)


class CausalConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, dilation):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel, dilation=dilation)

    def forward(self, x):
        return F.silu(self.conv(F.pad(x, (self.left, 0))))


class PrefixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv(1, 24, 9, 1), CausalConv(24, 32, 9, 4),
            CausalConv(32, 32, 9, 16), CausalConv(32, 32, 9, 32),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-5)
        raw = self.head(self.encoder(x).mean(-1))
        return F.softplus(raw[:, 0]), F.softplus(raw[:, 1]) + 1e-3


def conformal_quantile(scores, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires finite independent scores")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else float("inf")


def spectral_rate(x, fs):
    frequency, power = periodogram(x - np.mean(x), fs=fs)
    band = (frequency >= 0.1) & (frequency <= 0.7)
    selected = power[band]
    rate = 60 * frequency[band][np.argmax(selected)]
    concentration = selected.max() / max(selected.sum(), 1e-12)
    return float(rate), float(concentration)


def quality_gate(x, fs, min_cycles, concentration):
    rate, evidence = spectral_rate(x, fs)
    return bool(np.std(x) > 1e-6 and evidence >= concentration
                and len(x) / fs * rate / 60 >= min_cycles)


def reference_rate(events, time):
    completed = events[events <= time]
    if len(completed) < 2:
        return np.nan
    interval = completed[-1] - completed[-2]
    return 60 / interval if 1.4 <= interval <= 10 else np.nan
