"""Stage 2 of the pipeline: the feature cache (plan, "Encode and cache").

Each frozen encoder runs once per window, arm and channel. Features are stored as float16 in
``$PDMFM_DATA_ROOT/interim/features/<dataset>/<arm>-<encoder>-<key>/``:

* ``features.npy``: (N, C, K, D), memory-mapped; K = 6 (or 4) tokens for T, 1 otherwise;
* ``done.npy``: which windows are encoded; extraction resumes from it, so a job killed at
  its time limit continues where it stopped;
* ``meta.json``: the signature (window-store key, encoder name, checkpoint revision, arm),
  shapes and progress.

The key hashes the signature, so another checkpoint, another preprocessing version or other
windows give another folder: features are never mixed. Photos and texts have their own
caches (:func:`extract_images`, :func:`extract_texts`).

    python -m pdmfm.features extract --dataset cwru xjtu_sy --encoder mantis --arms T F E
    python -m pdmfm.features extract --dataset cwru --encoder moment --arms T --subsample 0.1
    python -m pdmfm.features images --dataset nonastreda najafi_thermal
    python -m pdmfm.features texts
    python -m pdmfm.features status
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pdmfm import arms, protocols
from pdmfm import texts as text_specs
from pdmfm.paths import data_root, interim_dir
from pdmfm.utils import save_npy_atomic, stable_hash, write_json_atomic
from pdmfm.windows import SPECS, WindowStore

FEATURES_VERSION = 1


def _hash(obj) -> str:
    return stable_hash(obj)[:12]


class FeatureSet:
    """One cached (dataset, encoder, arm): features of the windows marked done."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text(encoding="utf-8"))
        self.features = np.load(self.folder / "features.npy", mmap_mode="r")
        self.done = np.load(self.folder / "done.npy")

    @property
    def encoder(self) -> str:
        return self.meta["signature"]["encoder"]["encoder"]

    @property
    def arm(self) -> str:
        return self.meta["signature"]["arm"]

    def pooled(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), C * D) float32: tokens averaged, channels concatenated (Gate 1)."""
        block = np.asarray(self.features[np.asarray(rows)], dtype=np.float32)  # (n, C, K, D)
        return block.mean(axis=2).reshape(len(block), -1)


class FeatureStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else interim_dir("features")

    def signature(self, store: WindowStore, encoder_signature: dict, arm: str) -> dict:
        return {
            "windows": store.key,
            "dataset": store.dataset,
            "encoder": encoder_signature,
            "arm": arm,
            "features_version": FEATURES_VERSION,
        }

    def folder(self, store: WindowStore, encoder_signature: dict, arm: str) -> Path:
        signature = self.signature(store, encoder_signature, arm)
        return (
            self.root / store.dataset / f"{arm}-{encoder_signature['encoder']}-{_hash(signature)}"
        )

    def extract(
        self,
        store: WindowStore,
        encoder,
        arm: str,
        rows: np.ndarray | None = None,
        chunk: int = 256,
        log=print,
    ) -> FeatureSet:
        """Encode the windows in ``rows`` (a mask or indices; all by default) that are not
        cached yet, ``chunk`` windows at a time, saving progress after every chunk."""
        if (encoder.input_kind == "waveform") != (arm == "S"):
            raise ValueError(f"encoder {encoder.name} ({encoder.input_kind}) cannot take arm {arm}")
        if getattr(encoder, "input_rate", store.rate) != store.rate:
            raise ValueError(
                f"encoder {encoder.name} expects {encoder.input_rate} Hz, not {store.rate}"
            )
        encoder_signature = encoder.signature()  # once: it may look up the checkpoint revision
        signature = self.signature(store, encoder_signature, arm)
        folder = self.folder(store, encoder_signature, arm)
        n = len(store)
        shape = (n, len(arms.arm_channels(store, arm)), arms.n_tokens(store, arm), encoder.dim)
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["signature"] != signature or tuple(meta["shape"]) != shape:
                raise ValueError(f"{folder} holds other features; delete it to rebuild")
            features = np.load(folder / "features.npy", mmap_mode="r+")
            done = np.load(folder / "done.npy")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            features = np.lib.format.open_memmap(
                folder / "features.npy", mode="w+", dtype=np.float16, shape=shape
            )
            done = np.zeros(n, dtype=bool)
            save_npy_atomic(folder / "done.npy", done)
            meta = {
                "signature": signature,
                "shape": list(shape),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        wanted = np.zeros(n, dtype=bool)
        if rows is None:
            wanted[:] = True
        else:
            rows = np.asarray(rows)
            wanted[np.flatnonzero(rows) if rows.dtype == bool else rows] = True
        todo = np.flatnonzero(wanted & ~done)
        start_time = time.perf_counter()
        log(
            f"[features] {store.dataset} {arm} {encoder.name}: "
            f"{len(todo)} windows to encode, {int(done.sum())} cached"
        )
        for begin in range(0, len(todo), chunk):
            index = todo[begin : begin + chunk]
            inputs = arms.arm_input(store, index, arm)  # (m, C, inputs, length)
            m, c, k, length = inputs.shape
            embedded = encoder.encode(inputs.reshape(m * c * k, length)).reshape(m, c, k, -1)
            features[index] = arms.pool_tokens(embedded, arm).astype(np.float16)
            features.flush()
            done[index] = True
            save_npy_atomic(folder / "done.npy", done)
            meta.update(
                n_done=int(done.sum()),
                complete=bool(done.all()),
                updated=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            write_json_atomic(meta_path, meta)
            rate = (begin + len(index)) / max(time.perf_counter() - start_time, 1e-9)
            log(
                f"[features] {store.dataset} {arm} {encoder.name}: "
                f"{begin + len(index)}/{len(todo)} ({rate:.1f} windows/s)"
            )
        meta.update(n_done=int(done.sum()), complete=bool(done.all()))
        write_json_atomic(meta_path, meta)
        del features
        return FeatureSet(folder)

    def find(self, store: WindowStore, encoder: str, arm: str) -> FeatureSet:
        """The cache of ``encoder`` and ``arm`` built on this window store. With several
        (another checkpoint of the same encoder, e.g. a retrained PatchTST), the most
        recently updated complete one is used, with a warning naming the others."""
        matches = []
        for meta_path in (self.root / store.dataset).glob(f"{arm}-{encoder}-*/meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sig = meta["signature"]
            if (
                sig["windows"] == store.key
                and sig["encoder"]["encoder"] == encoder
                and sig["arm"] == arm
            ):
                recency = (
                    bool(meta.get("complete")),
                    meta.get("updated") or meta.get("created", ""),
                )
                matches.append((recency, meta_path.parent))
        if not matches:
            raise FileNotFoundError(
                f"no {encoder} {arm} features for {store.dataset} (key {store.key})"
            )
        matches.sort(key=lambda m: m[0])
        if len(matches) > 1:
            ignored = [folder.name for _, folder in matches[:-1]]
            warnings.warn(
                f"several {encoder} {arm} caches for {store.dataset}: using "
                f"{matches[-1][1].name}, ignoring {ignored} (delete them if stale)",
                stacklevel=2,
            )
        return FeatureSet(matches[-1][1])

    def status(self) -> pd.DataFrame:
        records = []
        for meta_path in sorted(self.root.glob("*/*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sig = meta.get("signature", {})
            total = (meta.get("shape") or [0])[0]
            complete = bool(meta.get("complete", False))
            records.append(
                {
                    "dataset": sig.get("dataset", meta_path.parent.parent.name),
                    "arm": sig.get("arm") or f"photo:{sig.get('kind')}",  # photo caches: kind
                    "encoder": (sig.get("encoder") or {}).get("encoder"),
                    "done": meta.get("n_done", total if complete else 0),
                    "windows": total,
                    "complete": complete,
                    "folder": meta_path.parent.name,
                }
            )
        return pd.DataFrame(records)


# ----------------------------------------------------------------------------- photos, texts


def image_paths(dataset: str) -> dict[str, list[str]]:
    """Every photo per kind, relative to the data root (attached or not)."""
    root = data_root()
    if dataset == "nonastreda":
        from pdmfm.datasets.nonastreda import PHOTO_FOLDERS
        from pdmfm.paths import raw_dir

        found = {}
        for kind, (folder, suffix) in PHOTO_FOLDERS.items():
            paths = sorted((raw_dir("nonastreda") / folder).glob(f"*{suffix}"))
            found[kind] = [p.relative_to(root).as_posix() for p in paths]
        return found
    if dataset == "najafi_thermal":
        from pdmfm.datasets import get_reader

        infos = get_reader("najafi_thermal").index()
        return {"thermal": [Path(i.meta["path"]).relative_to(root).as_posix() for i in infos]}
    if dataset == "matwi":
        from pdmfm.datasets.matwi import MATWI

        reader = MATWI()
        reader.prepare_crops()
        crops = sorted(reader.crops.rglob("*.jpg"))
        return {"insert": [p.relative_to(root).as_posix() for p in crops]}
    raise ValueError(f"{dataset} has no photos")


def extract_images(dataset: str, encoder, root: Path | None = None, log=print) -> list[Path]:
    """Photo tokens per kind: ``images/<dataset>-<kind>-<key>/{paths.csv, features.npy}``."""
    out_root = (Path(root) if root else interim_dir("features")) / "images"
    written = []
    for kind, paths in image_paths(dataset).items():
        if not paths:
            continue
        signature = {
            "dataset": dataset,
            "kind": kind,
            "encoder": encoder.signature(kind),
            "paths": _hash(paths),
        }
        folder = out_root / f"{dataset}-{kind}-{encoder.name}-{_hash(signature)}"
        if (folder / "meta.json").is_file():
            log(f"[images] {dataset} {kind}: cached in {folder.name}")
            written.append(folder)
            continue
        start = time.perf_counter()
        tokens = encoder.encode_paths([data_root() / p for p in paths], kind)
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": paths}).to_csv(folder / "paths.csv", index=False)
        save_npy_atomic(folder / "features.npy", tokens.astype(np.float16))
        write_json_atomic(
            folder / "meta.json",
            {"signature": signature, "shape": list(tokens.shape), "complete": True},
        )
        log(
            f"[images] {dataset} {kind}: {len(paths)} photos -> {tokens.shape} "
            f"in {time.perf_counter() - start:.0f} s"
        )
        written.append(folder)
    return written


def extract_image_views(
    dataset: str, encoder, views: int = 4, seed: int = 0, root: Path | None = None, log=print
) -> list[Path]:
    """Augmented copies of every photo: ``images/<dataset>-<kind>-<encoder>views-<key>/``
    with ``features.npy`` (views, n, tokens, dim) in the order of the photo cache's
    ``paths.csv``. Used only to augment training (pdmfm.model.inputs.PhotoViews)."""
    out_root = (Path(root) if root else interim_dir("features")) / "images"
    written = []
    for kind, paths in image_paths(dataset).items():
        if not paths:
            continue
        signature = {
            "dataset": dataset,
            "kind": kind,
            "encoder": encoder.signature(kind),
            "paths": _hash(paths),
            "views": views,
            "seed": seed,
            "augment": getattr(encoder, "AUGMENT", None),
        }
        folder = out_root / f"{dataset}-{kind}-{encoder.name}views-{_hash(signature)}"
        if (folder / "meta.json").is_file():
            log(f"[images] {dataset} {kind}: {views} views cached in {folder.name}")
            written.append(folder)
            continue
        start = time.perf_counter()
        files = [data_root() / p for p in paths]
        tokens = np.stack(
            [encoder.encode_paths(files, kind, augment_seed=(seed, v)) for v in range(1, views + 1)]
        )
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": paths}).to_csv(folder / "paths.csv", index=False)
        save_npy_atomic(folder / "features.npy", tokens.astype(np.float16))
        write_json_atomic(
            folder / "meta.json",
            {"signature": signature, "shape": list(tokens.shape), "complete": True},
        )
        log(
            f"[images] {dataset} {kind}: {views} augmented views of {len(paths)} photos "
            f"-> {tokens.shape} in {time.perf_counter() - start:.0f} s"
        )
        written.append(folder)
    return written


def extract_texts(
    encoder, datasets: list[str] | None = None, root: Path | None = None, log=print
) -> list[Path]:
    """Context embeddings (one per distinct sentence) and class prototypes (L2-normalized
    mean of each class's paraphrase embeddings) per dataset: ``texts/<dataset>-<key>.npz``."""
    out_root = (Path(root) if root else interim_dir("features")) / "texts"
    out_root.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset in datasets or text_specs.datasets():
        spec = text_specs.load(dataset)
        try:
            contexts = sorted(set(WindowStore.open(dataset).manifest["context"]))
        except FileNotFoundError:
            contexts = [spec.context_template] if "{" not in spec.context_template else []
        keys = list(spec.descriptions)
        paraphrases = [s for k in keys for s in spec.descriptions[k]]
        signature = {
            "dataset": dataset,
            "encoder": encoder.signature(),
            "contexts": contexts,
            "classes": spec.descriptions,
        }
        path = out_root / f"{dataset}-{encoder.name}-{_hash(signature)}.npz"
        if path.is_file():
            written.append(path)
            continue
        context_emb = (
            encoder.encode(contexts) if contexts else np.zeros((0, encoder.dim), np.float32)
        )
        prototypes = np.zeros((len(keys), encoder.dim), np.float32)
        if paraphrases:
            emb = encoder.encode(paraphrases)
            counts = [len(spec.descriptions[k]) for k in keys]
            for i, block in enumerate(np.split(emb, np.cumsum(counts)[:-1])):
                mean = block.mean(axis=0)
                prototypes[i] = mean / np.linalg.norm(mean)
        np.savez(
            path,
            contexts=np.array(contexts),
            context_emb=context_emb,
            classes=np.array(keys),
            prototypes=prototypes,
        )
        log(f"[texts] {dataset}: {len(contexts)} contexts, {len(keys)} prototypes -> {path.name}")
        written.append(path)
    return written


# ----------------------------------------------------------------------------- command line


def _claim(device: str, force: bool) -> None:
    if device == "cpu":
        from pdmfm.resources import claim_resources

        claim_resources("features", min_idle_cpus=2.0, min_free_ram_gb=2.0, force=force)


def _cmd_extract(args: argparse.Namespace) -> None:
    from pdmfm.encoders import get_encoder

    _claim(args.device, args.force)
    kwargs = {"batch_size": args.batch_size} if args.batch_size else {}
    encoder = get_encoder(args.encoder, device=args.device, checkpoint=args.checkpoint, **kwargs)
    feature_store = FeatureStore()
    for dataset in args.dataset:
        try:
            store = WindowStore.open(dataset)
        except (FileNotFoundError, ValueError) as error:  # e.g. Paderborn before its download
            print(f"[features] {dataset}: skipped ({error})")
            continue
        rows = protocols.subsample(store.manifest, args.subsample) if args.subsample else None
        for arm in args.arms:
            if arm in arms.unavailable(store):
                print(f"[features] {dataset}: no {arm} arm; skipped")
                continue
            feature_store.extract(store, encoder, arm, rows=rows, chunk=args.chunk)


def _cmd_images(args: argparse.Namespace) -> None:
    from pdmfm.encoders import DinoV2Encoder

    _claim(args.device, args.force)
    encoder = DinoV2Encoder(device=args.device)
    for dataset in args.dataset:
        extract_images(dataset, encoder)
        if args.views:
            extract_image_views(dataset, encoder, views=args.views)


def _cmd_texts(args: argparse.Namespace) -> None:
    from pdmfm.encoders import Qwen3TextEncoder

    _claim(args.device, args.force)
    extract_texts(Qwen3TextEncoder(device=args.device), args.dataset or None)


def _cmd_status(_: argparse.Namespace) -> None:
    table = FeatureStore().status()
    print(table.to_string(index=False) if len(table) else "no feature caches yet")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pdmfm.features", description=(__doc__ or "pdmfm").splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="encode window arms into the cache (resumable)")
    p.add_argument("--dataset", nargs="+", required=True, choices=sorted(SPECS))
    p.add_argument("--encoder", required=True, help="stats, mantis, moment, ast or patchtst")
    p.add_argument("--arms", nargs="+", default=["T", "F", "E"], choices=arms.ARMS)
    p.add_argument("--checkpoint", help="patchtst only: the model folder written by pdmfm.patchtst")
    p.add_argument("--subsample", type=float, help="unit-stratified fraction of windows, e.g. 0.1")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument(
        "--batch-size", type=int, default=0, help="encoder batch size (default per encoder)"
    )
    p.add_argument("--chunk", type=int, default=256, help="windows encoded between two saves")
    p.add_argument("--force", action="store_true", help="run even if the machine is busy")
    p.set_defaults(func=_cmd_extract)
    p = sub.add_parser("images", help="DINOv2 photo tokens")
    p.add_argument("--dataset", nargs="+", default=["nonastreda", "najafi_thermal"])
    p.add_argument("--views", type=int, default=0, help="augmented copies per photo (training)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_images)
    p = sub.add_parser("texts", help="Qwen3 context embeddings and class prototypes")
    p.add_argument("--dataset", nargs="*")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_texts)
    sub.add_parser("status", help="list the caches and their progress").set_defaults(
        func=_cmd_status
    )
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
