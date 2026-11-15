# pdmfm: multimodal predictive maintenance

A research pipeline for combining machine signals and inspection images with frozen
pretrained encoders, small fusion models, and calibrated uncertainty.

**Status: implementation and evaluation in progress.** The public CPU demo runs without
private repositories, datasets, or model downloads. Final target-adaptation experiments
and GPU performance measurements are pending; synthetic demo outputs are not model results.

## What is implemented

- Native readers for bearing and milling datasets, anti-aliased signal processing, causal
  photo alignment, and resumable feature caches.
- Per-modality evidential experts and concat, Perceiver, MBT, and C2F fusion candidates;
  text-description class prototypes and calibrated regression intervals.
- Unit-grouped evaluation, nested model selection, cross-dataset backbone pretraining,
  missing-modality checks, leakage baselines, and paired statistical reports.
- Causal RUL histories for XJTU-SY, Nonastreda, and MATWI; healthy-only anomaly scoring.
- Frequency-band explanations and physics checks where bearing geometry is supported.

```text
raw signals / photos -> window stores -> frozen encoder caches -> experts / fusion
                                                       |              |
                                                       +-> causal RUL +-> calibrated reports
                                                       +-> healthy-only anomaly scores
```

## Quick start

Python 3.11 or 3.12. From a fresh checkout:

```bash
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,eda]"
python -m pdmfm.demo --out runs/demo
pytest
```

Activate `.venv` before using `python`/`pytest`, or prefix those commands with `uv run`.
The demo creates synthetic MAT/CSV files and stand-in image/text features, trains small
heads on CPU, and writes classification, RUL, and anomaly reports under `runs/demo`.
Choose a fresh output folder for another run.

For pretrained encoding, install `.[encoders]`. MOMENT is an optional separate dependency;
its published dependency pins may require a separate extraction environment. Neither it
nor dataset downloads are needed for the default tests.

## Datasets and supported claims

| Dataset | Role | Task support and limitation |
| --- | --- | --- |
| Nonastreda | Source | Wear; healthy-only worn-tool novelty; RUL to a confirmed dulled endpoint on an estimated operating clock |
| MATWI | Source | Flank wear where the annotation is pure flank wear; novelty; remaining **recorded** operating life, a proxy endpoint |
| XJTU-SY | Source | Run-to-failure bearing RUL; anomaly onset assessed against derived FPT labels |
| CWRU | Source/probe | Seeded-fault diagnosis and healthy-reference novelty; load shift, not new-unit generalization; no RUL |
| Paderborn | Source/target bearing subsets | Diagnosis and novelty; real-file acceptance pending; no run-to-failure RUL |
| KAIST | Held-out target | Condition-shift diagnosis and novelty; asynchronous devices; no RUL |
| PHM 2010 | Held-out target | Cutter-wear transfer; real-file acceptance pending; further RUL protocol not implemented |
| Najafi / NASA milling / MBHM | Optional | Thermal classification / sparse wear regression / provenance-filtered pretraining research; outside the main evaluation |

[Dataset audit](docs/DATASET_AUDIT.md) explains endpoints, timing, missing labels, and
which conclusions were checked against downloaded files. All datasets retain their
publisher's licenses; the code's MIT license does not license data or checkpoints.

Set `PDMFM_DATA_ROOT` to keep large files outside the checkout; the default is ignored
`data/`. Set `HF_HOME` separately for pretrained weights. Environment helpers respect
caller-provided paths.

```bash
python -m pdmfm.fetch list
python -m pdmfm.fetch plan matwi
python -m pdmfm.lifecycle --dataset matwi
python -m pdmfm.lifecycle --dataset nonastreda --clock gap
python -m pdmfm.windows build nonastreda matwi
python -m pdmfm.features extract --dataset nonastreda matwi --encoder mantis --arms F
python -m pdmfm.features images --dataset nonastreda matwi
python -m pdmfm.features texts --dataset nonastreda matwi
python -m pdmfm.rul --dataset matwi --encoder mantis --arms F --run-id matwi-rul
python -m pdmfm.anomaly --dataset nonastreda --features mantis:F --run-id nona-anomaly
```

MATWI's revised reader invalidates older window stores. Rebuilding archives a changed
store instead of deleting it. Feature and experiment identities keep old and new protocols
separate. Nonastreda's historical 378-pass view remains for comparison; its RUL lifecycle
preserves all 512 passes and masks duplicate signals before their latest occurrence.

## Tests, compute, and research plan

- [Testing guide and requirement matrix](docs/TESTING.md): unit, integration, end-to-end,
  and opt-in acceptance suites. Expensive learning experiments run separately.
- [Current implementation plan](docs/PLAN.md) and [audit findings](docs/AUDIT.md).
- [CI](docs/CI.md) and [Ruche runbook](docs/RUCHE_RUNBOOK.md).
- [Current progress](PROJECT_LOG.md); historical notes are in `docs/archive/`.

Cached-feature training supports `--precision auto|fp32|bf16`, configurable prediction
batches and diagnostic frequency, and automatic GPU staging. Probability and likelihood
math retains FP32 accuracy under mixed precision. CUDA acceptance and a measured pilot
are required before claiming a speedup:

```bash
python -m pdmfm.benchmark --dataset nonastreda --encoder mantis --arms F --fits 120
```

The benchmark writes timings, peak allocation, numerical agreement, and a GPU-hour estimate.
It does not submit jobs. The experiment ceiling remains 110 A100-hours.

## Layout

`pdmfm/` contains readers, preprocessing, lifecycle conversion, encoders, training,
evaluation, and report tools. `tests/` separates test categories; `scripts/` contains
development and EDA helpers; `slurm/` contains cluster wrappers; `docs/` holds the current
specification, dataset evidence, and historical research notes.

Code: [MIT](LICENSE).
