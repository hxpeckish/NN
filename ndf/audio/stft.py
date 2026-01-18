import math
from typing import Any, Dict, Optional, Tuple

import torch


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WIN_LENGTH = int(0.032 * DEFAULT_SAMPLE_RATE)
DEFAULT_N_FFT = DEFAULT_WIN_LENGTH
DEFAULT_HOP_LENGTH = DEFAULT_WIN_LENGTH // 2


def _resolve_stft_params(config: Optional[Dict[str, Any]]) -> Tuple[int, int, int, bool]:
    if not config:
        return DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_WIN_LENGTH, True
    stft_cfg = config.get("stft", {})
    n_fft = int(stft_cfg.get("n_fft", DEFAULT_N_FFT))
    hop_length = int(stft_cfg.get("hop_length", DEFAULT_HOP_LENGTH))
    win_length = int(stft_cfg.get("win_length", DEFAULT_WIN_LENGTH))
    center = bool(stft_cfg.get("center", True))
    return n_fft, hop_length, win_length, center


def _sqrt_hann_window(win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    window = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    return torch.sqrt(window)


def stft(wav: torch.Tensor, config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    n_fft, hop_length, win_length, center = _resolve_stft_params(config)
    window = _sqrt_hann_window(win_length, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    return spec


def istft(spec: torch.Tensor, config: Optional[Dict[str, Any]] = None, length: Optional[int] = None) -> torch.Tensor:
    n_fft, hop_length, win_length, center = _resolve_stft_params(config)
    window = _sqrt_hann_window(win_length, device=spec.device, dtype=spec.dtype)
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )
    return wav
