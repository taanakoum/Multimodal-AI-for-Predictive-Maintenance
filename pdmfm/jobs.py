"""Portable one-seed job manifests and isolated packed GPU workers for Slurm arrays.

Every task owns a distinct run directory. No array worker writes another fold's summary.
The manifest is immutable; logs and completion records live beside it in a task folder.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pdmfm import protocols
from pdmfm.paths import REPO_ROOT
from pdmfm.utils import stable_hash, write_json_atomic

STAGES = ("planb", "pretrain", "rul", "gate2", "tune")


def make(
    stage,
    run_id,
    datasets,
    folds,
    encoder="mantis",
    arms=("F",),
    seed=0,
    gate1=None,
    decisions=(),
    init=None,
    epochs=None,
    batch_size=None,
):
    if stage not in STAGES or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("unknown stage or invalid run ID")
    choices = None
    if gate1:
        report = json.loads(Path(gate1).read_text(encoding="utf-8"))
        if not report.get("complete"):
            raise ValueError("Gate 1 is incomplete; finish source probes before generating jobs")
        choices = report["per_fold_winner"]
        if any(str(fold) not in choices for fold in folds):
            raise ValueError("Gate 1 has no eligible shared winner for a requested fold")
    if not datasets or not set(datasets) <= set(protocols.SOURCE_FOLDS):
        raise ValueError("job datasets must be source datasets with grouped outer folds")
    if len(set(datasets)) != len(datasets) or not folds or len(set(folds)) != len(folds):
        raise ValueError("datasets and folds must be nonempty and unique")
    if (epochs is not None and epochs < 1) or (batch_size is not None and batch_size < 1):
        raise ValueError("epochs and batch size must be positive")
    if stage == "rul" and "paderborn" in datasets:
        raise ValueError("Paderborn has no run-to-failure labels for RUL")
    tasks = []
    for fold in folds:
        protocols._check_fold(fold)
        name, selected_arms = encoder, tuple(arms)
        if choices is not None:
            from pdmfm.probe import parse_candidate

            name, selected_arms = parse_candidate(choices[str(fold)])
        name = name.replace("{fold}", str(fold))
        for dataset in datasets if stage != "pretrain" else ["shared"]:
            task_id = f"{run_id}-{dataset}-fold{fold}"
            command = [
                "python",
                "-m",
                f"pdmfm.{stage}",
                "--encoder",
                name,
                "--arms",
                *selected_arms,
                "--run-id",
                task_id,
                "--folds",
                str(fold),
                "--device",
                "cuda",
            ]
            if stage == "pretrain":
                # CWRU contributes the fixed condition-shift source probe/pretraining split.
                command += ["--datasets", "cwru", *datasets]
            else:
                command += ["--dataset", dataset]
            command += (
                ["--seed", str(seed)] if stage in ("gate2", "tune") else ["--seeds", str(seed)]
            )
            if stage == "tune":
                command += ["--grid", "lr=0.0003,0.001"]
            if epochs is not None:
                command += ["--epochs", str(epochs)]
            if batch_size is not None:
                if stage not in ("planb", "pretrain", "rul"):
                    raise ValueError("batch-size override is supported for planb/pretrain/rul")
                command += ["--batch-size", str(batch_size)]
            if decisions:
                if stage not in ("pretrain", "planb"):
                    raise ValueError("decisions apply only to pretrain and planb")
                files = [p.format(dataset=dataset, fold=fold, seed=seed) for p in decisions]
                if not all(Path(p).is_file() for p in files):
                    raise FileNotFoundError("a fold-specific decision file is missing")
                command += ["--per-fold", *files]
            if init:
                if stage != "planb":
                    raise ValueError("initialization is supported by planb only")
                checkpoint = init.format(fold=fold, seed=seed)
                if not Path(checkpoint).is_file():
                    raise FileNotFoundError(f"pretraining checkpoint is missing: {checkpoint}")
                command += ["--init", checkpoint]
            tasks.append({"id": task_id, "fold": fold, "dataset": dataset, "argv": command})
    return {
        "version": 1,
        "stage": stage,
        "seed": seed,
        "folds": list(folds),
        "protocol": protocols.SIGNATURE,
        "tasks": tasks,
    }


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("manifest tasks must be nonempty and unique")
    for task in tasks:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", task["id"]):
            raise ValueError("invalid task ID")
    return manifest


def collect(path, dataset, out, run_root=None):
    """Pool all five completed outer folds after array jobs exit, with provenance."""
    manifest = read_manifest(path)
    stage = manifest["stage"]
    if stage not in ("planb", "rul"):
        raise ValueError("collection supports final planb/rul evaluation only")
    tasks = [t for t in manifest["tasks"] if t["dataset"] == dataset]
    if sorted(t["fold"] for t in tasks) != list(range(1, 6)):
        raise ValueError("collection requires all five outer folds for one dataset")
    root = Path(run_root) if run_root else REPO_ROOT / "runs"
    sources = []
    for task in tasks:
        status = Path(path).with_suffix("") / f"{task['id']}.json"
        state = json.loads(status.read_text()) if status.is_file() else {}
        if state.get("returncode") != 0:
            raise ValueError(f"incomplete array task: {task['id']}")
        if state.get("identity") != stable_hash({"manifest": manifest, "task": task}):
            raise ValueError(f"task manifest changed: {task['id']}")
        stem = f"fold{task['fold']}-seed{manifest['seed']}"
        source = root / stage / task["id"]
        for suffix in (".json", "-bags.csv"):
            result = source / f"{stem}{suffix}"
            if not result.is_file():
                raise FileNotFoundError(result)
            sources.append(result)
    out = Path(out)
    identity = {"manifest": manifest, "dataset": dataset}
    marker = out / "collection.json"
    if out.exists() and any(out.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError("output is not this manifest's collection; use a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(marker, identity)
    for source in sources:
        shutil.copy2(source, out / source.name)
    from pdmfm.planb import summarize

    summarize(out, title=f"{stage}: {dataset}, five outer folds, seed {manifest['seed']}")
    return out


def launch(path, index=0, workers=2, require_cuda=True):
    """Run a shard; workers share the single GPU granted to this array element.

    ``require_cuda=False`` is for subprocess orchestration tests only. Production CLI
    always requires one visible GPU and preserves Slurm's CUDA_VISIBLE_DEVICES mask.
    """
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    manifest = read_manifest(path)
    chunks = math.ceil(len(manifest["tasks"]) / workers)
    if not 0 <= index < chunks:
        raise ValueError("array index outside manifest")
    if require_cuda:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("packed array elements require exactly one visible CUDA GPU")
    selected = manifest["tasks"][index * workers : (index + 1) * workers]
    if require_cuda:
        # Fast structural guard against load_sources silently dropping unavailable sources.
        # Full numerical scans belong in the CPU preflight before submission.
        from pdmfm.features import FeatureStore
        from pdmfm.windows import WindowStore

        for task in selected:
            argv = task["argv"]
            datasets = [task["dataset"]]
            if manifest["stage"] == "pretrain":
                tail = argv[argv.index("--datasets") + 1 :]
                datasets = tail[
                    : next((i for i, t in enumerate(tail) if t.startswith("--")), len(tail))
                ]
            encoder = argv[argv.index("--encoder") + 1]
            tail = argv[argv.index("--arms") + 1 :]
            selected_arms = tail[: next(i for i, t in enumerate(tail) if t.startswith("--"))]
            from pdmfm.arms import unavailable

            for dataset in datasets:
                store = WindowStore.open(dataset)
                protocols.audit_folds(dataset, store.manifest)
                applicable = [a for a in selected_arms if a not in unavailable(store)]
                if not applicable:
                    raise ValueError(f"{dataset}: no applicable arm in the selected candidate")
                for arm in applicable:
                    if not FeatureStore().find(store, encoder, arm).done.all():
                        raise ValueError(f"{dataset}: incomplete {encoder}:{arm} cache")
    folder = Path(path).with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) // len(selected))

    def one(task):
        identity = stable_hash({"manifest": manifest, "task": task})
        status = folder / f"{task['id']}.json"
        if status.exists():
            previous = json.loads(status.read_text(encoding="utf-8"))
            if previous["identity"] != identity:
                raise ValueError("task settings changed; generate a fresh manifest filename")
            if previous["returncode"] == 0:
                return 0
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "PDMFM_MAX_THREADS": str(threads),
            "PDMFM_GPU_MEMORY_FRACTION": str(0.9 / len(selected)),
        }
        argv = list(task["argv"])
        if argv[0] == "python":
            argv[0] = sys.executable
        started = time.perf_counter()
        with (folder / f"{task['id']}.out").open("a", encoding="utf-8") as output:
            result = subprocess.run(argv, env=env, stdout=output, stderr=subprocess.STDOUT)
        write_json_atomic(
            status,
            {
                "identity": identity,
                "returncode": result.returncode,
                "seconds": time.perf_counter() - started,
                "threads": threads,
                "memory_fraction": 0.9 / len(selected),
            },
        )
        print(f"{task['id']}: exit {result.returncode}", flush=True)
        return result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        outcomes = list(pool.map(one, selected))
    return 1 if any(outcomes) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    make_p = sub.add_parser("make")
    make_p.add_argument("--stage", choices=STAGES, required=True)
    make_p.add_argument("--run-id", required=True)
    make_p.add_argument("--datasets", nargs="+", required=True)
    make_p.add_argument("--folds", nargs="+", type=int, default=list(range(1, 6)))
    make_p.add_argument("--encoder", default="mantis")
    make_p.add_argument("--arms", nargs="+", default=["F"])
    make_p.add_argument("--seed", type=int, default=0)
    make_p.add_argument("--gate1", type=Path)
    make_p.add_argument("--decisions", nargs="+", default=[])
    make_p.add_argument("--init")
    make_p.add_argument("--epochs", type=int)
    make_p.add_argument("--batch-size", type=int)
    make_p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("collect")
    q.add_argument("manifest", type=Path)
    q.add_argument("--dataset", required=True)
    q.add_argument("--out", type=Path, required=True)
    for action in ("count", "run", "show"):
        q = sub.add_parser(action)
        q.add_argument("manifest", type=Path)
        q.add_argument("--workers", type=int, choices=range(1, 9), default=2)
        if action == "run":
            q.add_argument(
                "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
            )
    args = p.parse_args(argv)
    if args.action == "make":
        manifest = make(
            args.stage,
            args.run_id,
            args.datasets,
            args.folds,
            args.encoder,
            args.arms,
            args.seed,
            args.gate1,
            args.decisions,
            args.init,
            args.epochs,
            args.batch_size,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists() and json.loads(args.out.read_text(encoding="utf-8")) != manifest:
            raise ValueError("manifest exists with other settings; use a new filename")
        write_json_atomic(args.out, manifest)
        print(f"{len(manifest['tasks'])} tasks, one seed ({args.seed}): {args.out}")
    elif args.action == "count":
        print(math.ceil(len(read_manifest(args.manifest)["tasks"]) / args.workers))
    elif args.action == "show":
        for task in read_manifest(args.manifest)["tasks"]:
            print(shlex.join(task["argv"]))
    elif args.action == "collect":
        print(collect(args.manifest, args.dataset, args.out))
    else:
        raise SystemExit(launch(args.manifest, args.index, args.workers))


if __name__ == "__main__":
    main()
