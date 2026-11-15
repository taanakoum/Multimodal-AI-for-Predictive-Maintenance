"""Precision and cached-input staging shared by training and extraction."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast(device, precision: str = "auto"):
    device = torch.device(device)
    if precision not in ("auto", "fp32", "bf16"):
        raise ValueError("precision must be auto, fp32 or bf16")
    supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "bf16" and not supported:
        raise ValueError("bf16 requires a CUDA device with BF16 support")
    if supported and precision != "fp32":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class InputBank:
    """Stage immutable inputs once; keep at most half the free device memory occupied.

    Alternative photo views still use the normal batch path. CPU staging is pinned on
    CUDA machines so transfers can run asynchronously. No global cache owns GPU tensors.
    """

    def __init__(self, inputs, device, policy="auto"):
        self.device = torch.device(device)
        if policy not in ("auto", "cpu", "cuda"):
            raise ValueError("cache_device must be auto, cpu or cuda")
        self.parts = {}
        size = 0
        for name, m in inputs.modalities.items():
            part = {"tokens": m.tokens, "present": m.present}
            for key in ("token_present", "levels", "age"):
                value = getattr(m, key)
                if value is not None:
                    part[key] = value
            self.parts[name] = {k: torch.as_tensor(v.copy()) for k, v in part.items()}
            size += sum(v.numel() * v.element_size() for v in self.parts[name].values())
        resident = policy == "cuda"
        if self.device.type == "cuda" and policy == "auto":
            resident = size <= torch.cuda.mem_get_info(self.device)[0] // 2
        if resident and self.device.type != "cuda":
            raise ValueError("cache_device=cuda requires a CUDA training device")
        for part in self.parts.values():
            for key, value in part.items():
                if resident:
                    part[key] = value.to(self.device)
                elif self.device.type == "cuda":
                    part[key] = value.pin_memory()

    def batch(self, rows, drop=()):
        result = {}
        for name, part in self.parts.items():
            index = torch.as_tensor(rows, device=part["tokens"].device)
            result[name] = {k: v[index].to(self.device, non_blocking=True) for k, v in part.items()}
            if name in drop:
                result[name]["present"] = torch.zeros_like(result[name]["present"])
        return result
