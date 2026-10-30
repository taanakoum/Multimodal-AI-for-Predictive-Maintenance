"""Frozen encoders behind one interface (plan stage 2, "Encode and cache").

Sensor encoders map encoder inputs to vectors: ``encode(x)`` takes (m, 512) arm inputs
(``input_kind = "series"``) or (m, L) raw windows at the model rate (``"waveform"``, AST) and
returns (m, dim) float32. ``signature()`` names the checkpoint and its revision; it is part
of every feature-cache key, so features of different checkpoints are never mixed.

| Name | Checkpoint | dim | Input |
| --- | --- | --- | --- |
| ``stats`` | none (handcrafted: 16 band means and maxima, mean, std, min, max) | 36 | series |
| ``mantis`` | paris-noah/Mantis-8M | 256 | series |
| ``moment`` | AutonLab/MOMENT-1-base | 768 | series |
| ``patchtst`` | own checkpoint (``pdmfm.patchtst``) | d_model | series |
| ``ast`` | MIT/ast-finetuned-audioset-10-10-0.4593, 100 frames | 768 | waveform |

Image (DINOv2-S/14) and text (Qwen3-Embedding-0.6B) encoders have their own interfaces.
Heavy libraries are imported only when an encoder is created.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from pdmfm import preprocess as pp


def hf_revision(repo: str) -> str | None:
    """Commit hash of the cached snapshot of ``repo`` (None if it is not cached)."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - not cached, offline, or no hub installed
        return None


def _batches(x: np.ndarray, size: int):
    for start in range(0, len(x), size):
        yield x[start : start + size]


def _on_device(model, device):
    """Move a torch module to ``device`` (in place) in evaluation mode. ``torch.nn.Module.to``
    is called directly because some libraries' own ``.to`` signatures do not type-check."""
    import torch

    torch.nn.Module.to(model, device)
    return model.eval()


def _torch_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device)


class Encoder:
    name = "encoder"
    dim = 0
    input_kind = "series"
    repo: str | None = None

    def signature(self) -> dict:
        return {"encoder": self.name, "dim": self.dim, "repo": self.repo, "revision": self.revision}

    @property
    def revision(self) -> str | None:
        return hf_revision(self.repo) if self.repo else None

    def encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class StatsEncoder(Encoder):
    """Handcrafted reference, no weights: the mean and the maximum of 16 equal bands of the
    512-vector, then its mean, std, min and max. On F and E inputs the band maxima keep
    narrow lines (a defect line dominates one bin and barely moves its band's mean)."""

    name = "stats"
    dim = 36

    def __init__(self, device: str = "cpu", batch_size: int = 0) -> None:
        del device, batch_size

    def signature(self) -> dict:
        return {"encoder": self.name, "dim": self.dim, "version": 2}

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        bands = x.reshape(len(x), 16, -1)
        moments = np.stack([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)
        return np.concatenate([bands.mean(axis=2), bands.max(axis=2), moments], axis=1).astype(
            np.float32
        )


class MantisEncoder(Encoder):
    name = "mantis"
    dim = 256
    repo = "paris-noah/Mantis-8M"

    def __init__(self, device: str = "cpu", batch_size: int = 256) -> None:
        from mantis.architecture import MantisV1  # pyright: ignore[reportMissingImports]
        from mantis.trainer import MantisTrainer  # pyright: ignore[reportMissingImports]

        _torch_device(device)
        network = MantisV1(device=device).from_pretrained(self.repo)
        self.model = MantisTrainer(device=device, network=network)
        self.batch_size = batch_size

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        parts = [
            np.asarray(self.model.transform(b[:, None, :])) for b in _batches(x, self.batch_size)
        ]
        return np.concatenate(parts).astype(np.float32)


class MomentEncoder(Encoder):
    name = "moment"
    dim = 768
    repo = "AutonLab/MOMENT-1-base"

    def __init__(self, device: str = "cpu", batch_size: int = 64) -> None:
        from momentfm import MOMENTPipeline  # pyright: ignore[reportMissingImports]

        self.device = _torch_device(device)
        model = MOMENTPipeline.from_pretrained(
            str(self.repo), model_kwargs={"task_name": "embedding"}
        )
        model.init()
        self.model = _on_device(model, self.device)
        self.batch_size = batch_size

    def encode(self, x: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for b in _batches(np.asarray(x, dtype=np.float32), self.batch_size):
                batch = torch.from_numpy(b[:, None, :]).to(self.device)
                embeddings = torch.as_tensor(self.model(x_enc=batch).embeddings)
                out.append(embeddings.float().cpu().numpy())
        return np.concatenate(out)


class PatchTSTEncoder(Encoder):
    """Own PatchTST from :mod:`pdmfm.patchtst`; the patch embeddings are averaged."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", batch_size: int = 512) -> None:
        from transformers import PatchTSTModel  # pyright: ignore[reportMissingImports]

        self.checkpoint = Path(checkpoint)
        self.device = _torch_device(device)
        model = PatchTSTModel.from_pretrained(self.checkpoint, do_mask_input=False)
        self.model = _on_device(model, self.device)
        self.dim = int(model.config.d_model)
        self.name = f"patchtst-{self.checkpoint.parent.name}"  # e.g. patchtst-fold1
        self.batch_size = batch_size

    def signature(self) -> dict:
        weights = sorted(self.checkpoint.glob("*.safetensors")) or sorted(
            self.checkpoint.glob("*.bin")
        )
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in weights)).hexdigest()[:16]
        return {"encoder": self.name, "dim": self.dim, "checkpoint": digest}

    def encode(self, x: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for b in _batches(np.asarray(x, dtype=np.float32), self.batch_size):
                values = torch.from_numpy(b[:, :, None]).to(self.device)  # (m, 512, 1 channel)
                hidden = self.model(past_values=values).last_hidden_state  # (m, 1, patches, d)
                out.append(hidden.mean(dim=(1, 2)).float().cpu().numpy())
        return np.concatenate(out)


def resize_ast_positions(model, max_frames: int):
    """An AST copy for inputs of ``max_frames`` frames: the time axis of the position
    embedding grid is interpolated (bilinear). The pretrained model pads every input to
    1024 frames (10.24 s), so a 1 s window would be 90 % padding."""
    import copy

    import torch
    from transformers import ASTModel  # pyright: ignore[reportMissingImports]

    config = copy.deepcopy(model.config)
    config.max_length = max_frames
    resized = ASTModel(config)
    f_old, t_old = model.embeddings.get_shape(model.config)
    f_new, t_new = resized.embeddings.get_shape(config)
    state = model.state_dict()
    positions = state["embeddings.position_embeddings"]  # (1, 2 + f*t, hidden)
    hidden = positions.shape[-1]
    grid = positions[:, 2:].reshape(1, f_old, t_old, hidden).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(
        grid, size=(f_new, t_new), mode="bilinear", align_corners=False
    )
    state["embeddings.position_embeddings"] = torch.cat(
        [positions[:, :2], grid.permute(0, 2, 3, 1).reshape(1, f_new * t_new, hidden)], dim=1
    )
    resized.load_state_dict(state)
    return resized.eval()


class ASTEncoder(Encoder):
    """S arm: raw windows resampled to 16 kHz, log-mel features, AST pooled output."""

    name = "ast"
    dim = 768
    input_kind = "waveform"
    repo = "MIT/ast-finetuned-audioset-10-10-0.4593"
    AUDIO_RATE = 16_000.0

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 32,
        max_frames: int = 100,
        input_rate: float = pp.MODEL_RATE_HZ,
    ) -> None:
        from transformers import ASTFeatureExtractor, ASTModel  # pyright: ignore[reportMissingImports]

        self.device = _torch_device(device)
        self.extractor = ASTFeatureExtractor.from_pretrained(str(self.repo), max_length=max_frames)
        model = ASTModel.from_pretrained(str(self.repo))
        if max_frames != model.config.max_length:
            model = resize_ast_positions(model, max_frames)
        self.model = _on_device(model, self.device)
        self.max_frames, self.input_rate, self.batch_size = max_frames, input_rate, batch_size

    def signature(self) -> dict:
        return {**super().signature(), "max_frames": self.max_frames, "input_rate": self.input_rate}

    def encode(self, x: np.ndarray) -> np.ndarray:
        import torch

        wave = pp.resample(np.asarray(x, dtype=np.float32), self.input_rate, self.AUDIO_RATE)
        out = []
        with torch.no_grad():
            for b in _batches(wave, self.batch_size):
                features = self.extractor(
                    list(b), sampling_rate=int(self.AUDIO_RATE), return_tensors="pt"
                )
                values = features["input_values"].to(self.device)
                out.append(self.model(input_values=values).pooler_output.float().cpu().numpy())
        return np.concatenate(out)


SERIES_ENCODERS = {
    "stats": StatsEncoder,
    "mantis": MantisEncoder,
    "moment": MomentEncoder,
    "ast": ASTEncoder,
}
# Encoder cost order: breaks Gate-1 ties in favour of the cheaper encoder.
COST = {"stats": 0, "mantis": 1, "patchtst": 2, "moment": 3, "ast": 4}


def get_encoder(name: str, device: str = "cpu", checkpoint: str | Path | None = None, **kwargs):
    if name == "patchtst" or name.startswith("patchtst-"):
        if checkpoint is None:
            raise ValueError("patchtst needs --checkpoint (a folder written by pdmfm.patchtst)")
        return PatchTSTEncoder(checkpoint, device=device, **kwargs)
    if name not in SERIES_ENCODERS:
        raise KeyError(
            f"unknown encoder {name!r}; choose from {sorted(SERIES_ENCODERS)} or patchtst"
        )
    return SERIES_ENCODERS[name](device=device, **kwargs)


def cost(name: str) -> int:
    return COST.get(name.split("-")[0], 9)


# ----------------------------------------------------------------------------- images, text


class DinoV2Encoder:
    """DINOv2-S/14 photo tokens: class token + patch grid average-pooled to a small grid.

    Tool photos (3.1:1) are resized to 224x672 and pooled to 2x4 (9 tokens); chip, workpiece
    and thermal images to 224x224 and 2x2 (5 tokens)."""

    name = "dinov2"
    dim = 384
    repo = "facebook/dinov2-small"
    SIZE = {"tool": (224, 672), "chip": (224, 224), "workpiece": (224, 224), "thermal": (224, 224)}
    GRID = {"tool": (2, 4), "chip": (2, 2), "workpiece": (2, 2), "thermal": (2, 2)}

    # DINOv2's processor normalizes with the ImageNet statistics; using them directly avoids
    # needing its preprocessor config in an offline cache.
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __init__(self, device: str = "cpu", batch_size: int = 16) -> None:
        from transformers import AutoModel  # pyright: ignore[reportMissingImports]

        self.device = _torch_device(device)
        self.model = _on_device(AutoModel.from_pretrained(self.repo), self.device)
        self.batch_size = batch_size

    def signature(self, kind: str) -> dict:
        return {
            "encoder": self.name,
            "repo": self.repo,
            "revision": hf_revision(self.repo),
            "size": self.SIZE[kind],
            "grid": self.GRID[kind],
        }

    def n_tokens(self, kind: str) -> int:
        rows, cols = self.GRID[kind]
        return 1 + rows * cols

    # Augmented copies (fix 2 of the image plan, PROJECT_LOG week 5): a crop keeping 80-100 %
    # of each side, brightness and contrast within ±15 %, and mirroring for chips and
    # workpieces only (a mirrored end mill is not a real tool view).
    AUGMENT = {"crop": (0.8, 1.0), "jitter": 0.15, "mirror": ("chip", "workpiece", "thermal")}

    def _pixels(self, path: Path, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
        from PIL import Image, ImageEnhance, ImageOps

        height, width = self.SIZE[kind]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if rng is not None:
                low, high = self.AUGMENT["crop"]
                w, h = image.size
                cw, ch = int(w * rng.uniform(low, high)), int(h * rng.uniform(low, high))
                left, top = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
                image = image.crop((left, top, left + cw, top + ch))
                jitter = self.AUGMENT["jitter"]
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - jitter, 1 + jitter))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(1 - jitter, 1 + jitter))
                if kind in self.AUGMENT["mirror"] and rng.random() < 0.5:
                    image = ImageOps.mirror(image)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (array - self.MEAN) / self.STD

    def encode_paths(
        self, paths: list[Path], kind: str, augment_seed: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """Tokens of each photo; with ``augment_seed``, of an augmented copy that depends
        only on the seed and the photo's position."""
        import torch

        height, width = self.SIZE[kind]
        grid_h, grid_w = height // 14, width // 14
        out = []
        with torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                chunk = range(start, min(start + self.batch_size, len(paths)))
                pixels = np.stack(
                    [
                        self._pixels(
                            paths[i],
                            kind,
                            None
                            if augment_seed is None
                            else np.random.default_rng((*augment_seed, i)),
                        )
                        for i in chunk
                    ]
                )
                hidden = self.model(
                    pixel_values=torch.from_numpy(pixels).to(self.device)
                ).last_hidden_state
                cls, patches = hidden[:, :1], hidden[:, 1 : 1 + grid_h * grid_w]
                patches = patches.reshape(len(pixels), grid_h, grid_w, -1).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.adaptive_avg_pool2d(patches, self.GRID[kind])
                pooled = pooled.flatten(2).transpose(1, 2)  # (n, rows*cols, 384)
                out.append(torch.cat([cls, pooled], dim=1).float().cpu().numpy())
        return np.concatenate(out)


class Qwen3TextEncoder:
    name = "qwen3_embedding"
    dim = 1024
    repo = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

        _torch_device(device)
        # float32: the checkpoint's bfloat16 is slow on CPU and leaves norms off by ~2e-3.
        self.model = SentenceTransformer(
            self.repo, device=device, model_kwargs={"dtype": torch.float32}
        )

    def signature(self) -> dict:
        return {
            "encoder": self.name,
            "repo": self.repo,
            "revision": hf_revision(self.repo),
            "dtype": "float32",
        }

    def encode(self, sentences: list[str]) -> np.ndarray:
        z = np.asarray(self.model.encode(sentences), dtype=np.float32)
        return z / np.linalg.norm(z, axis=1, keepdims=True)
