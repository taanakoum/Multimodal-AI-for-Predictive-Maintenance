# pdmfm: Project 14 multimodal predictive-maintenance model

Project 14 builds a multimodal predictive-maintenance model from frozen pretrained encoders:
- a sensor foundation model, picked at Gate 1;
- DINOv2 for photos and thermal images;
- Qwen3-Embedding for text.

The encoders' outputs are cached as tokens and fused by a small trained stage, chosen by a fusion search. Evidential heads, whose classes are text descriptions of faults, give each prediction a calibrated uncertainty and a trust value per modality.

- **Plan:** [Project 14 implementation plan](https://claude.ai/artifact/1M2DPuffJP1mRt2oiUujx5) (reference copy in `docs/PLAN.md`)
- **Progress, findings and decisions:** [`PROJECT_LOG.md`](PROJECT_LOG.md)
- **Earlier work reused here:** the OmniFuser reproduction in `../OmniFuser`, installed as an editable dependency

## Setup (Windows development machine)

Large files never live in this OneDrive folder: data, weights and the environment are on `D:\Project14`.

```powershell
uv venv D:\Project14\.venv --python 3.11
. .\scripts\env\dev.ps1                   # sets PDMFM_DATA_ROOT, HF_HOME, activates the venv
uv pip install -e ".[dev]" -e ..\OmniFuser
uv pip install -e ".[encoders,forecast,xai]" "momentfm @ git+https://github.com/moment-timeseries-foundation-model/moment.git"
pytest
```

`momentfm` comes from GitHub because its PyPI release (0.1.4) pins transformers 4.33.3 and numpy 1.25.2.

## Data

```powershell
python -m pdmfm.fetch list                # registered datasets, roles, licences
python -m pdmfm.fetch plan kaist          # files, sizes and publisher hashes; downloads nothing
python -m pdmfm.fetch get kaist           # resumable; verifies size and SHA-256
python -m pdmfm.fetch extract kaist       # zip/rar archives into extracted/kaist
```

Downloads land in `$PDMFM_DATA_ROOT/raw/<dataset>`, each with a `MANIFEST.json` that records the URL, size and SHA-256 of every file. `python -m pdmfm.fetch verify <dataset>` re-hashes a folder against its manifest and records files placed by hand.

Two datasets need steps only you can take. [`docs/USER_ACTIONS.md`](docs/USER_ACTIONS.md) has the commands and the acceptance tests for each:
- **Paderborn:** its host refuses connections from this machine; fetch it on Ruche or through a VPN.
- **PHM 2010:** the original download host is gone; get the zips from a mirror and verify them.

## Exploratory data analysis

```powershell
python scripts\eda\run_eda.py             # every dataset on disk; checks CPU and RAM first
```

The findings and their consequences for preprocessing, splits and baselines are in [`docs/EDA_REPORT.md`](docs/EDA_REPORT.md). Every sensor channel goes through `pdmfm.preprocess.model_input`: resampled to 12 kHz, then band-limited to 5 kHz, because recorders' anti-alias roll-offs above 5 kHz identify the source, and on CWRU even the class.

## Tests and CI

```powershell
pytest                                    # real-data tests run when the data is on disk, skip otherwise
bash scripts/ci/run_local_ci.sh           # CI reproduced in a fresh environment (Git Bash, uv)
pre-commit install                        # same lint and format checks before each commit
```

GitHub Actions runs lint, then tests on Ubuntu and Windows with Python 3.11 and 3.12, then an advisory type check. A weekly workflow checks that the data sources still serve the files we used. See [`docs/CI.md`](docs/CI.md).

## Pipeline (stages 1-5 of the plan)

```powershell
python -m pdmfm.windows build cwru kaist xjtu_sy nonastreda matwi   # 1: window stores (model input, labels, contexts)
python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda matwi --encoder mantis --arms T F E
python -m pdmfm.features images --dataset nonastreda matwi --views 4   # 2: DINOv2 photo tokens (MATWI: insert crops)
python -m pdmfm.features texts                                   #    Qwen3 contexts and class prototypes
python -m pdmfm.patchtst --fold 1 --device cuda                  #    own PatchTST, per outer fold (Ruche)
python -m pdmfm.probe --run-id gate1 --candidates mantis:T mantis:T+E moment:T ...   # 3: Gate 1
python -m pdmfm.tune --dataset nonastreda --encoder mantis --arms F --run-id tune-nona --grid lr=3e-4,1e-3 d=128,256
python -m pdmfm.planb --dataset nonastreda --encoder mantis --arms F --run-id planb-nona     # 4: Plan B
python -m pdmfm.gate2 --dataset nonastreda --encoder mantis --arms F --run-id gate2-nona     # 5: Gate 2
python -m pdmfm.rul --encoder mantis --arms F E --run-id rul-xjtu --compare runs/planb/<run>  #    RUL from past snapshots
python -m pdmfm.features status                                  # what is cached
```

- **Stage 1.** A window store per dataset holds the band-limited model input of every channel,
  level descriptors, slow-channel summaries, the measured shaft speed, labels and the context
  sentence. Milling windows (Nonastreda, MATWI) also get the previous pass's or cut's photos.
- **Stage 2.** Feature caches are keyed by the store, the encoder checkpoint and the arm, so
  they are never mixed, and they resume after an interruption.
- **Splits.** All splits are by physical unit (`pdmfm.protocols`).
- **Stage 3.** Gate 1 ranks encoder-and-arm candidates over its four tasks on
  inner-validation units only.
- **Stage 4.** Plan B, the baseline and fallback model: one evidential expert per modality
  (force, vibration, current, slow channels, photos), with text-anchored class heads and NIG
  regression heads. Experts are combined by Dempster's rule (classes) and MoNIG (wear, RUL),
  and calibrated on inner-validation units (temperature; interval scale). It runs over 5
  outer folds × 3 seeds, with per-modality, missing-modality, stale-photo and baseline
  results, the fit on the training units, and pooled out-of-fold estimates with 95 %
  intervals from resampling test units (`runs/planb/<run-id>/summary.md`).
  `pdmfm.tune` chooses hyperparameters on the inner-validation units only. Every table also
  has a schedule baseline (the class and target at the same pass, cut or snapshot number) and,
  for the wear classes, ordinal metrics.
- **Stage 5 (Gate 2).** Four fusion candidates (concat, Perceiver, MBT, C2F) on Plan B's
  experts, ranked against Plan B by inner cross-validation, with validation rotated over the
  training units (`pdmfm.gate2`). RUL gets a causal model over each bearing's past snapshots
  (`pdmfm.rul`).
- **Ruche.** MOMENT, AST and PatchTST at full scale, and the full Gate 2, run there: [`docs/RUCHE_RUNBOOK.md`](docs/RUCHE_RUNBOOK.md).

## Heavy jobs and Ruche

Scripts that load the CPU call `pdmfm.resources.claim_resources` first, which is aware of SLURM
allocations. `python -m pdmfm.resources` shows idle CPUs, free RAM and busy processes. On Ruche,
`slurm/` holds:

- the environment (`env.sh`);
- the job wrappers (`job.sbatch` for GPU, `cpu.sbatch`, `checks.sbatch`);
- `chain.sh` for runs beyond 24 h;
- `status.sh`;
- the login-node downloads.

The runbook has the exact commands.

## Layout

```text
pdmfm/            package
  paths.py        data root and folders (PDMFM_DATA_ROOT)
  sources.py      dataset registry: URLs, hashes, licences, split lists
  fetch.py        resumable, hash-checked downloader; extract; verify
  schema.py       ImageItem and the as-of join of photos to windows
  preprocess.py   resampling, 5 kHz band limit, windows, T/F/E arms, descriptors, FPT, RUL
  resources.py    CPU/RAM guard and thread caps (SLURM-aware)
  datasets/       one reader per dataset
  texts/          context templates and fault descriptions per dataset (YAML)
  protocols.py    unit-level splits: outer folds, Gate 1, targets, few-shot draws, pretraining masks
  windows.py      stage 1: window stores
  arms.py         encoder inputs T, F, E, S from a window store
  encoders.py     frozen encoders (stats, Mantis, MOMENT, PatchTST, AST, DINOv2, Qwen3)
  features.py     stage 2: resumable feature caches; photo and text caches
  patchtst.py     own PatchTST, masked-patch pretraining per outer fold
  probe.py        stage 3: Gate 1
  losses.py       evidential rules and losses: Dempster's rule (TMC), Dirichlet loss, NIG loss, MoNIG
  metrics.py      accuracy, macro-F1, ECE, Brier, NLL, temperature scaling, interval coverage, PHM 2012 score
  model/          per-modality token sets from the caches (inputs.py), encoders and heads (layers.py), Plan B (late.py), Gate-2 candidates (fusion.py)
  train.py        supervised training on cached features: losses, checkpoint selection, bag aggregation
  planb.py        stage 4: Plan B over outer folds and seeds; summary tables
  tune.py         hyperparameter search on inner-validation units
  gate2.py        stage 5: fusion search by inner cross-validation, and the Gate-2 rule
  rul.py          RUL from a causal history of snapshots (XJTU-SY)
  stats.py        pooled out-of-fold statistics: macro-F1, RMSE, unit bootstrap intervals
tests/            pytest suite (synthetic data; real-data, checkpoint, acceptance and network tests are marked)
scripts/eda/      exploratory data analysis
scripts/ci/       local CI reproduction, data-source check
scripts/env/      environment scripts for the dev machine
slurm/            Ruche: environment, job wrappers, chains, status, network check, downloads
docs/             plan copy, EDA report, Ruche runbook, CI, your action list
```

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences, listed by `python -m pdmfm.fetch list`. Paderborn is CC BY-NC 4.0, so model weights trained on it are for non-commercial use only.
