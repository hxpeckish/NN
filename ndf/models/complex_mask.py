import torch


def apply_complex_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.shape != ref.shape:
        raise ValueError(f"mask shape {mask.shape} must match ref shape {ref.shape}")
    if mask.size(-1) != 2:
        raise ValueError("mask/ref last dimension must be 2 for real/imag")
    mask_real, mask_imag = mask[..., 0], mask[..., 1]
    ref_real, ref_imag = ref[..., 0], ref[..., 1]
    out_real = mask_real * ref_real - mask_imag * ref_imag
    out_imag = mask_real * ref_imag + mask_imag * ref_real
    return torch.stack([out_real, out_imag], dim=-1)
