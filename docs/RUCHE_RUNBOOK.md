# Ruche: current execution guide

The specification is [PLAN.md](PLAN.md); the older installation notes and detailed cluster
examples remain in [archive](archive/RUCHE_RUNBOOK-previous.md). Dataset/weights directories
live in `$WORKDIR`; no private sibling checkout is needed.

## Setup and CPU checks

Configure the module/environment names in `slurm/env.sh` for your account. Prefetch model
weights on a network-enabled login node with `slurm/prefetch_weights.sh`. Jobs use
`HF_HUB_OFFLINE=1`. The Python environment must contain the package and required extras.

```bash
mkdir -p runs/logs
sbatch slurm/checks.sbatch
sbatch slurm/cpu.sbatch python -m pdmfm.windows build nonastreda matwi
sbatch slurm/cpu.sbatch python -m pdmfm.lifecycle --dataset matwi
sbatch slurm/cpu.sbatch python -m pdmfm.lifecycle --dataset nonastreda --clock gap
```

Inspect `pdmfm.fetch plan` before dataset downloads. Paderborn/PHM 2010 still require real
files and acceptance (`pytest -m data --run-data`), rather than assuming synthetic tests
prove their physical channel layouts. Revised MATWI stores use a new signature; rebuilding
archives a changed store. Large raw files and historical experiment outputs are preserved.

## GPU validation, extraction, and budget

```bash
sbatch slurm/job.sbatch python -m pytest -m gpu --run-gpu
sbatch slurm/job.sbatch python -m pdmfm.features extract --dataset nonastreda matwi --encoder mantis --arms F --device cuda
sbatch slurm/job.sbatch python -m pdmfm.features images --dataset nonastreda matwi --device cuda --precision auto
sbatch slurm/job.sbatch python -m pdmfm.benchmark --dataset nonastreda --encoder mantis --arms F --fits 120
```

Sensor/image caches resume completed chunks. Mantis' numpy extraction stays FP32. Small
fusion fits use automatic BF16 where supported, retaining FP32 likelihood math. The
benchmark records elapsed time, peak allocation and inference agreement. Its fit-count
estimate is not permission to exceed the 110 A100-hour total experiment ceiling. Add
extraction/pretraining/other-stage pilot costs when maintaining the full job ledger.

## Fold-safe source runs

Run Gate 1 on source validation units only, then Gate 2 per source dataset. Gate 2 defaults
to a ten-epoch screen; only its two finalists and both late references receive the full
inner-fold evaluation. `--screen-epochs 0` requests the exhaustive search.

Use Gate 1's `report.json` through `pretrain --gate1` and source tune/Gate-2 decisions through
`pretrain --per-fold`. For a backbone shared across datasets, only unanimous source choices
override common defaults; disagreeing fusion choices fall back to late experts. This rule
is fold-local and independent of file order. Do not substitute a pooled decision for an
outer fold's own decision. Outer-fold PatchTST caches are refused by nested searches until
inner-split pretraining is implemented.

```bash
python -m pdmfm.pretrain --encoder mantis --arms F --run-id source-pretrain --gate1 runs/probe/gate1/report.json --per-fold runs/gate2/nona/decision.json runs/gate2/matwi/decision.json --device cuda
python -m pdmfm.planb --dataset nonastreda --encoder mantis --arms F --run-id nona-finetune --init 'runs/pretrain/source-pretrain/fold{fold}-seed{seed}/best.pt' --device cuda
python -m pdmfm.rul --dataset matwi --encoder mantis --arms F --run-id matwi-rul --device cuda
python -m pdmfm.anomaly --dataset nonastreda --features mantis:F --run-id nona-anomaly
```

Choose the encoder/arms matching each fold's backbone when fine-tuning. A backbone with
another encoder, architecture or training-unit provenance is rejected. Target adaptation,
LoRA, forecasting and final held-out evaluations remain future research stages.
