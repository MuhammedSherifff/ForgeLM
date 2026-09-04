import torch

from utils import resolve_amp_dtype


class FakeCuda:
    """Monkeypatch target for torch.cuda capability/support queries."""

    def __init__(self, major: int, bf16_supported: bool):
        self.major = major
        self.bf16_supported = bf16_supported


def test_bf16_selected_on_capable_hardware(monkeypatch):
    fake = FakeCuda(major=8, bf16_supported=True)

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (fake.major, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_amp_dtype(torch.device("cuda")) is torch.bfloat16


def test_fp16_on_pre_ampere_even_when_emulation_reported(monkeypatch):
    # T4-class GPU: PyTorch may report emulated bf16 support, but using it
    # silently breaks fused SDPA kernels and exhausts VRAM.
    fake = FakeCuda(major=7, bf16_supported=True)

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (fake.major, 5)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_amp_dtype(torch.device("cuda")) is torch.float16


def test_fp16_when_bf16_unsupported(monkeypatch):
    fake = FakeCuda(major=8, bf16_supported=False)

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device: (fake.major, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert resolve_amp_dtype(torch.device("cuda")) is torch.float16


def test_cpu_returns_float16_fallback():
    # AMP is disabled on CPU by the caller; the helper's contract is to
    # return fp16 for any non-CUDA device.
    assert resolve_amp_dtype(torch.device("cpu")) is torch.float16
