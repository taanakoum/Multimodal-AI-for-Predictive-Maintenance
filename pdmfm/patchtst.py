"""Own PatchTST (plan, pretraining level 1): masked-patch pretraining on vibration segments.

For outer fold k the corpus is every vibration channel of the windows that
:func:`pdmfm.protocols.pretraining_mask` allows (no target dataset, none of fold k's test or
inner-validation units), cut into the 512-sample T-arm segments. The model follows the plan:
d = 128, 3 layers, 16 heads, patches of 16 without overlap, per-input scaling (RevIN-like);
40 % of the patches are masked and the loss is the MSE on them (PatchTST, section 3.2).

Training saves ``last.pt`` (model, optimizer, schedule, epoch) after every epoch, so a job
stopped at its time limit resumes, and the model with the lowest held-out loss in
``model/``, which :class:`pdmfm.encoders.PatchTSTEncoder` reads. A resume with other
settings or another corpus is refused.

    python -m pdmfm.patchtst --fold 1 --device cuda          # -> runs/patchtst/fold1/
    python -m pdmfm.patchtst --fold 1 --epochs 1 --max-segments 2000   # CPU smoke test
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from pdmfm import preprocess as pp
from pdmfm import protocols
from pdmfm.paths import REPO_ROOT
from pdmfm.utils import write_json_atomic
from pdmfm.windows import WindowStore

CORPUS_DATASETS = ("cwru", "xjtu_sy", "paderborn")
MODEL = {
    "num_input_channels": 1,
    "context_length": pp.SEGMENT,
    "patch_length": 16,
    "patch_stride": 16,
    "d_model": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 16,
    "ffn_dim": 256,
    "dropout": 0.1,
    "do_mask_input": True,
    "mask_type": "random",
    "random_mask_ratio": 0.4,
    "scaling": "std",
}


def corpus(
    fold: int | None,
    datasets: tuple[str, ...] = CORPUS_DATASETS,
    window_root: Path | None = None,
    max_segments: int | None = None,
    seed: int = 0,
    log=print,
) -> tuple[np.ndarray, dict]:
    """(n, 512) float32 vibration segments allowed for fold ``fold``, and counts per dataset.
    With ``max_segments``, windows are sampled first, so only what is kept is read."""
    rng = np.random.default_rng(seed)
    plan = []
    for dataset in datasets:
        try:
            store = WindowStore.open(dataset, window_root)
        except FileNotFoundError:
            log(f"[patchtst] {dataset}: no window store; left out of the corpus")
            continue
        rows = np.flatnonzero(protocols.pretraining_mask(dataset, store.manifest, fold))
        vibration = [i for i, f in enumerate(store.families) if f == "vibration"]
        plan.append((dataset, store, rows, vibration, store.window // pp.SEGMENT))
    if not plan:
        raise FileNotFoundError("no window store of " + ", ".join(datasets) + "; build them first")
    total = sum(len(rows) * len(vib) * seg for _, _, rows, vib, seg in plan)
    keep = 1.0 if not max_segments or total <= max_segments else max_segments / total
    parts, counts = [], {}
    for dataset, store, rows, vibration, segments in plan:
        if not len(rows) or not vibration:  # nothing this fold may use
            counts[dataset] = {"windows": 0, "segments": 0}
            continue
        if keep < 1.0:
            rows = np.sort(rng.choice(rows, max(1, math.ceil(keep * len(rows))), replace=False))
        for begin in range(0, len(rows), 512):
            block = np.asarray(store.signals[rows[begin : begin + 512]][:, vibration])
            block = block[:, :, : segments * pp.SEGMENT].reshape(-1, pp.SEGMENT)
            parts.append(block[block.std(axis=1) > 1e-8])  # flat segments carry nothing
        counts[dataset] = {
            "windows": int(len(rows)),
            "segments": int(len(rows) * len(vibration) * segments),
        }
    if not parts:
        raise ValueError(f"fold {fold}: no vibration segments may be used for pretraining")
    data = np.concatenate(parts)  # float32 already: no second copy
    if max_segments and len(data) > max_segments:
        data = data[rng.choice(len(data), max_segments, replace=False)]
    return data, counts


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _save_model(model, folder: Path) -> None:
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def pretrain(
    out_dir: Path,
    fold: int | None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = "cpu",
    max_segments: int | None = None,
    seed: int = 0,
    val_fraction: float = 0.05,
    window_root: Path | None = None,
    model_overrides: dict | None = None,
    log=print,
) -> dict:
    import torch
    from transformers import PatchTSTConfig, PatchTSTForPretraining

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    data, counts = corpus(
        fold, window_root=window_root, max_segments=max_segments, seed=seed, log=log
    )
    order = np.random.default_rng(seed).permutation(len(data))
    n_val = max(1, int(val_fraction * len(data)))
    val, train = torch.from_numpy(data[order[:n_val]]), torch.from_numpy(data[order[n_val:]])
    settings = {
        "fold": fold,
        "model": {**MODEL, **(model_overrides or {})},
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "max_segments": max_segments,
        "n_train": len(train),
        "n_val": len(val),
        "corpus": counts,
    }
    device_ = torch.device(device)
    if device_.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = PatchTSTForPretraining(PatchTSTConfig(**settings["model"])).to(device_)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total, warmup = epochs * steps_per_epoch, max(1, int(0.05 * epochs * steps_per_epoch))

    def schedule(step: int) -> float:  # linear warmup, then cosine to zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    state_path = out_dir / "last.pt"
    start_epoch, best, history = 0, float("inf"), []
    if state_path.is_file():
        state = torch.load(state_path, map_location=device_, weights_only=False)
        if state["settings"] != settings:
            raise ValueError(f"{out_dir} was started with other settings or data; use a new --out")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        if device_.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best, history = state["epoch"], state["best"], state["history"]
        log(f"[patchtst] resuming after epoch {start_epoch} (best held-out loss {best:.4f})")
    log(f"[patchtst] fold {fold}: {len(train)} training and {len(val)} held-out segments; {counts}")
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        permutation = torch.randperm(len(train), generator=generator)
        running, tick = 0.0, time.perf_counter()
        for step in range(steps_per_epoch):
            batch = train[permutation[step * batch_size : (step + 1) * batch_size]]
            loss = model(past_values=batch[:, :, None].to(device_)).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch)
        model.eval()
        # The same masks every epoch make held-out losses comparable; fork_rng keeps that
        # seeding from repeating the training masks of every epoch.
        devices = [device_] if device_.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            held_out = sum(
                float(model(past_values=val[i : i + batch_size, :, None].to(device_)).loss)
                * len(val[i : i + batch_size])
                for i in range(0, len(val), batch_size)
            ) / len(val)
        history.append(
            {
                "epoch": epoch + 1,
                "train": running / len(train),
                "held_out": held_out,
                "seconds": time.perf_counter() - tick,
            }
        )
        if held_out < best:
            best = held_out
            _save_model(model, out_dir / "model")
        _atomic_torch_save(
            {
                "settings": settings,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "history": history,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device_.type == "cuda" else None,
            },
            state_path,
        )
        progress = {"fold": fold, "epoch": epoch + 1, "epochs": epochs, "best_held_out": best}
        write_json_atomic(out_dir / "progress.json", progress)  # read by slurm/status.sh
        log(
            f"[patchtst] epoch {epoch + 1}/{epochs}: train {history[-1]['train']:.4f}, "
            f"held-out {held_out:.4f}, {history[-1]['seconds']:.0f} s"
        )
    metrics = {
        **settings,
        "best_held_out": best,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    write_json_atomic(out_dir / "metrics.json", metrics)
    log(f"[patchtst] done: best held-out loss {best:.4f} -> {out_dir / 'model'}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.patchtst", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--fold", type=int, help="outer fold 1-5; omit only for smoke tests (all source units)"
    )
    parser.add_argument(
        "--out", type=Path, help="default: runs/patchtst/fold<k> (or runs/patchtst/all)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-segments", type=int, help="random subset of the corpus (smoke tests)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="run on a busy machine (CPU only)")
    args = parser.parse_args(argv)
    if args.fold is not None and not 1 <= args.fold <= protocols.N_FOLDS:
        parser.error(f"--fold must be 1..{protocols.N_FOLDS}")
    if args.device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("patchtst", min_idle_cpus=2.0, min_free_ram_gb=3.0, force=args.force)
    name = f"fold{args.fold}" if args.fold is not None else "all"
    out = args.out or REPO_ROOT / "runs" / "patchtst" / name
    pretrain(
        out, args.fold, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, max_segments=args.max_segments, seed=args.seed,
    )  # fmt: skip


if __name__ == "__main__":
    main()
