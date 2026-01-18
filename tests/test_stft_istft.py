import torch

from ndf.audio.stft import istft, stft


def test_stft_istft_roundtrip():
    torch.manual_seed(0)
    wav = torch.randn(2, 16000)
    spec = stft(wav)
    recon = istft(spec, length=wav.shape[-1])
    error = torch.mean(torch.abs(wav - recon)).item()
    assert error < 1e-2
