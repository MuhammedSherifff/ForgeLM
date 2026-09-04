"""Small shared runtime helpers."""

import torch


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    """Pick the autocast dtype actually supported by the hardware.

    bf16 requires compute capability >= 8.0. Newer PyTorch reports
    ``is_bf16_supported() == True`` on older GPUs via software emulation,
    but emulated bf16 silently disables the fused SDPA kernels and forces
    the math fallback, which materializes full attention matrices and
    exhausts VRAM. Only trust bf16 on capable hardware; everything else
    gets fp16 (with GradScaler).
    """
    if (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 8
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16
