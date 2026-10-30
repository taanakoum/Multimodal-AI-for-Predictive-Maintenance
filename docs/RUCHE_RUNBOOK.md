# Ruche runbook: encoder caches, PatchTST pretraining, Gate 1 and Plan B

This runbook finishes stages 2-3 of the plan on **Ruche**, the cluster of the Mésocentre
Paris-Saclay:

- the feature caches that are too slow for the laptop (MOMENT, AST, your own PatchTST);
- the Paderborn data, which only Ruche can download;
- the Gate-1 encoder probe on all four of its tasks;
- the Plan B baseline (stage 4) with the Gate-1 winner, on Nonastreda, XJTU-SY and Paderborn.

It follows the structure and the conventions of the OmniFuser runbook
(`../OmniFuser/TRAINING_RUNBOOK_RUCHE.md`):

- Python from a conda module, with `source activate`;
- PyTorch built for CUDA 12.6, so it runs on both the V100s and the A100s;
- jobs with `--export=NONE` that load `slurm/env.sh`;
- `gpua100` for real runs.

If you have never used a cluster, read §1 and §8 of that runbook first: they explain login
nodes, jobs, partitions and fairshare. Ruche facts come from the Ruche course (November 2023);
when something differs, the current user documentation wins:
<https://mesocentre.pages.centralesupelec.fr/user_doc/>.

Commands in `powershell` blocks run **on your laptop**; commands in `bash` blocks run **on
Ruche** (in the SSH session).

**Contents**

| Part | Sections |
|---|---|
| Overview | What runs where · the jobs · before you start · the path at a glance |
| A: Set up (once) | §1 Laptop · §2 Connect · §3 Storage · §4 Code · §5 Data · §6 Python environment · §7 `slurm/env.sh` · §8 Weights · §9 SLURM scripts |
| B: The runs | §10 Every session · §11 Step 0: checks · §12 Step 1: Paderborn · §13 Step 2: feature caches · §14 Step 3: PatchTST · §15 Step 4: Gate 1 · §16 Step 5: Plan B · §17 Step 6: download |
| C: Operating | §18 Watch, resume, cancel · §19 Troubleshooting · §20 Rules · §21 Cheat sheet |

## Overview

### What runs where

| Work | Where | Why |
|---|---|---|
| Window stores of CWRU, KAIST, XJTU-SY, Nonastreda (`python -m pdmfm.windows build`) | Laptop (done) | The data is there |
| Mantis features, DINOv2 photo tokens, Qwen3 texts, a first Gate-1 probe | Laptop (done; `PROJECT_LOG.md`) | Fast enough on the CPU |
| Paderborn download | Ruche login node | Its host refuses the laptop |
| Paderborn extraction, acceptance tests, window store | Ruche CPU job | Needs only a RAR tool |
| MOMENT (arms T, F, E) and AST (arm S) on all windows | Ruche GPU | About 40 h each on the laptop CPU; minutes on an A100 |
| Own PatchTST pretraining (5 folds), then its features | Ruche GPU | Pretraining needs a GPU |
| Gate-1 probe on all four tasks | Ruche CPU job (or the laptop) | Logistic and ridge regressions |
| Plan B on Nonastreda and XJTU-SY with Mantis and the handcrafted reference | Laptop (done; `PROJECT_LOG.md`, week 5) | About 70 s per fold and seed on the CPU |
| Plan B with the Gate-1 winner, and on Paderborn | Ruche GPU | Needs the full caches and the Paderborn store; minutes on an A100 |

### The jobs

| Step | What | Partition | Expected time | Output |
|---|---|---|---|---|
| 0 | Tests, stores and caches present; GPU check; network check | `cpu_short`, `gpu_test` | 10-20 min | `runs/logs/` |
| 1 | Paderborn: fetch (login node), extract, acceptance tests, window store | login, `cpu_short` | 1-3 h download, under 1 h job | `$PDMFM_DATA_ROOT/interim/windows/paderborn` |
| 2 | Feature caches: handcrafted (CPU), Mantis, MOMENT, AST (GPU) | `cpu_short`, `gpua100` | Minutes to an hour each | `$PDMFM_DATA_ROOT/interim/features/` |
| 3 | PatchTST pretraining, one model per outer fold, then its features | `gpua100` | Under an hour per fold | `runs/patchtst/fold1..5/` |
| 4 | Gate-1 probe | `cpu_short` | Under 1 h | `runs/gate1/gate1-v1/` |
| 5 | Plan B: 5 outer folds × 3 seeds per dataset, with the Gate-1 winner and the handcrafted reference | `gpua100` | Minutes to an hour per job | `runs/planb/planb-v1-*/` |
| 6 | Download caches and results | laptop | Minutes | `D:\Project14\data\interim\features`, `runs/` |

The times are estimates. The first job of each kind tells you the real ones (§18).

### Before you start

| You need | Notes |
|---|---|
| A Ruche account | As for OmniFuser |
| This repository pushed to GitHub | `git push` after your commits; Ruche clones it (§4) |
| A GitHub token with read access, or an SSH key on GitHub | The repository is private, so `git clone` asks for credentials (§4) |
| The four window stores on the laptop | `python -m pdmfm.windows info` lists cwru, kaist, nonastreda and xjtu_sy |
| About 40 GB free in `$WORKDIR` | Environment about 6 GB, weights about 4 GB, window stores about 2.5 GB, Paderborn several GB raw plus its store, caches and runs a few GB |

### The path at a glance

1. **Laptop:** push the commits; pack the window stores (§1).
2. **Connect** (§2), prepare storage (§3), clone (§4), upload the stores (§5).
3. **Install** the environment (§6), set `slurm/env.sh` (§7), download the weights (§8).
4. **Every session:** the §10 block.
5. **Step 0** checks (§11), then **Paderborn** (§12).
6. **Feature caches** (§13) and **PatchTST** (§14), in parallel.
7. **Gate 1** (§15) when steps 1-3 are complete.
8. **Plan B** (§16) with the Gate-1 winner; then **download** (§17).

---

# Part A: Set up (once)

## 1. On your laptop

Push your work (Ruche clones what GitHub has):

```powershell
cd "C:\Users\ali_a\OneDrive\Desktop\Multimodal AI for Predictive Maintenance"
git status -sb
git push
git log -1 --oneline
```

`git status -sb` must show `## master...origin/master` with nothing after it; note the commit
the last line prints.

Pack the window stores, with the photo tokens and the text prototypes. The stores are built
from your local data and are identical on Ruche, so their keys and the feature caches built
on them match on both machines. Plan B needs the photo tokens (the raw photos never go to
Ruche) and the class prototypes; both are small and keyed by paths relative to the data
root, so they work there as they are:

```powershell
cd D:\Project14\data
tar -czf pdmfm-windows.tar.gz interim/windows/cwru interim/windows/xjtu_sy interim/windows/nonastreda interim/windows/kaist interim/features/images interim/features/texts
```

The archive is about 2.3 GB: the signals are float32 and compress little. KAIST is a target,
used only from stage 8; leave it out if the upload is slow.

## 2. Connect

```powershell
$RucheLogin = "your-ruche-login"
$Ruche = "$RucheLogin@ruche.mesocentre.universite-paris-saclay.fr"
ssh $Ruche
```

Define the two variables in every new PowerShell window; §5 and §17 use them. The OmniFuser
runbook (§3) shows how to log in with an SSH key instead of a password.

## 3. Storage

| Directory | Variable | Limit | Use |
|---|---|---|---|
| `/home/<login>` | `$HOME` | 50 GB | Small configuration files |
| `/workdir/<login>` | `$WORKDIR` | 500 GB | Code, data, conda environment, results |

Neither is backed up. Conda installs into `$HOME/.conda` by default, which fills the home
quota. If you did not already move it for OmniFuser:

```bash
if [ -d "$HOME/.conda" ] && [ ! -L "$HOME/.conda" ] && [ ! -e "$WORKDIR/.conda" ]; then mv "$HOME/.conda" "$WORKDIR/.conda"; fi
mkdir -p "$WORKDIR/.conda"
if [ ! -e "$HOME/.conda" ]; then ln -s "$WORKDIR/.conda" "$HOME/.conda"; fi
ls -ld "$HOME/.conda"
```

The last line must show `.conda -> /workdir/<login>/.conda`.

## 4. Get the code

```bash
cd "$WORKDIR"
git clone https://github.com/taanakoum/Multimodal-AI-for-Predictive-Maintenance.git pdmfm
cd pdmfm
git log -1 --oneline
```

- The repository is private. When `git clone` asks, give your GitHub user name and, as the
  password, a token: GitHub → Settings → Developer settings → Fine-grained tokens, with
  read-only *Contents* access to this repository.
- The last line must show the commit you noted in §1.
- Everything from now on happens inside `$WORKDIR/pdmfm`.
- Later changes: `git pull`. Never pull while a job runs.
- **If the clone fails** (no GitHub access from Ruche), send an archive instead:

  ```powershell
  git archive --format=tar.gz --prefix=pdmfm/ -o pdmfm-code.tar.gz HEAD
  scp pdmfm-code.tar.gz "${Ruche}:/workdir/$RucheLogin/"
  ```

  Then on Ruche: `cd "$WORKDIR" && tar -xzf pdmfm-code.tar.gz && cd pdmfm`.

## 5. Upload the window stores

```powershell
scp D:\Project14\data\pdmfm-windows.tar.gz "${Ruche}:/workdir/$RucheLogin/"
```

```bash
mkdir -p "$WORKDIR/pdmfm_data"
tar -xzf "$WORKDIR/pdmfm-windows.tar.gz" -C "$WORKDIR/pdmfm_data" && rm "$WORKDIR/pdmfm-windows.tar.gz"
ls "$WORKDIR/pdmfm_data/interim/windows"
```

The `ls` must list `cwru kaist nonastreda xjtu_sy` (without `kaist` if you left it out), and
`ls "$WORKDIR/pdmfm_data/interim/features"` must list `images texts`.

## 6. Python environment

**6.1 The conda module.** List the conda modules, then load the newest `anaconda3` or
`miniconda3` (the name below is an example):

```bash
module avail anaconda miniconda
module load anaconda3/2023.09-0/none-none
```

**6.2 Create the environment and install the project** (10-20 minutes; allowed on the login
node):

```bash
cd "$WORKDIR/pdmfm"
conda create -n pdmfm python=3.11 -y
source activate pdmfm
python --version
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -e ".[encoders,eda,dev]"
python -m pip install --no-cache-dir "momentfm @ git+https://github.com/moment-timeseries-foundation-model/moment.git"
```

| Command | What it does |
|---|---|
| `conda create -n pdmfm python=3.11 -y` | An environment named `pdmfm` with Python 3.11 (the project needs 3.11 or 3.12) |
| `source activate pdmfm` | Switches to it. Ruche asks for `source activate`; never `conda activate` or `conda init`, which edit `~/.bashrc` |
| `pip install torch --index-url …/cu126` | PyTorch for CUDA 12.6: it has code for the V100s (`gpu_test`, `gpu`) and the A100s (`gpua100`) |
| `pip install -e ".[encoders,eda,dev]"` | This project with Transformers, PEFT, sentence-transformers, Mantis, matplotlib and the test tools; keeps the PyTorch above |
| `pip install "momentfm @ git+…"` | MOMENT from GitHub: its PyPI release pins outdated libraries (`PROJECT_LOG.md`, week 1) |

`--no-cache-dir` keeps pip's cache out of the home quota. If `conda create` fails with
`PackagesNotFoundError`, add `-c conda-forge`.

OmniFuser is **not** needed on Ruche: the Nonastreda window store comes from the laptop.

## 7. Tell the job scripts about the environment

Open `slurm/env.sh` (`nano slurm/env.sh`; save with Ctrl+O, Enter; quit with Ctrl+X) and set
the two names from §6:

```bash
CONDA_MODULE="anaconda3/2023.09-0/none-none"
CONDA_ENV="pdmfm"
```

Every job loads the environment through this file. It also sets:

- `PDMFM_DATA_ROOT=$WORKDIR/pdmfm_data`;
- `HF_HOME=$WORKDIR/hf_cache`;
- `HF_HUB_OFFLINE=1` inside jobs.

Check it:

```bash
source slurm/env.sh
python -c "import torch, transformers, momentfm, mantis, pdmfm; print(torch.__version__, transformers.__version__)"
python -m pdmfm.windows info
```

The first line prints two versions (the torch one ends in `+cu126`). The second lists the
uploaded stores with their keys, which must equal those `python -m pdmfm.windows info` prints
on the laptop.

## 8. Download the weights (login node, once)

Compute nodes may have no internet access, so the encoder checkpoints are downloaded here,
into `$HF_HOME` (about 4 GB):

```bash
source slurm/env.sh
bash slurm/prefetch_weights.sh
```

It prints one `ok <repo> <path>` line per checkpoint: Mantis, MOMENT, DINOv2, Qwen3 and AST.
Step 0 (§11) then checks that every checkpoint loads offline on a compute node.

## 9. The scripts in `slurm/`

| File | Role |
|---|---|
| `slurm/env.sh` | Loads the conda module and the environment; sets the data root, the weight cache and offline mode (§7) |
| `slurm/job.sbatch` | Runs **one command** on one GPU (`gpua100`, 8 CPUs, 24 h unless you override). Whatever follows `slurm/job.sbatch` is the command |
| `slurm/cpu.sbatch` | The same on a CPU node (`cpu_short`, 8 CPUs, 1 h) |
| `slurm/checks.sbatch` | Step 0 on a CPU node |
| `slurm/chain.sh` | Submits the same job several times in a row, for work beyond 24 h (every step here resumes) |
| `slurm/status.sh` | Jobs, stores, caches, PatchTST and Gate-1 runs, logs with errors, disk use. Reads files only (login node OK) |
| `slurm/net_check.sbatch` | Which download hosts a node reaches |
| `slurm/prefetch_weights.sh`, `slurm/fetch_paderborn.sh` | Login-node downloads |

Options on the `sbatch` line go **before** the script name, the command after it. For
example, `--partition=gpu_test --time=00:30:00` runs the same job on the test partition for at
most 30 minutes. Partitions (from the OmniFuser runbook):

| Partition | Hardware per node | Max time | Use |
|---|---|---|---|
| `gpu_test` | 4 × V100 32 GB | 1 h, one job at a time | GPU check, smoke tests |
| `gpua100` | 4 × A100 40 GB | 24 h; at most 4 GPUs per user | All real GPU runs (default of `slurm/job.sbatch`) |
| `gpu` | 4 × V100 32 GB | 24 h | If `gpua100` is crowded |
| `cpu_short` | CPU only | 1 h, one job at a time | Checks, Paderborn, Gate 1 |

Keep every encoder whose features are compared on the **same GPU type**. Results on V100 and
A100 differ slightly.

---

# Part B: The runs

## 10. Start of every session

```bash
cd "$WORKDIR/pdmfm"
mkdir -p runs/logs
source slurm/env.sh
bash slurm/status.sh
```

`mkdir -p runs/logs` matters: SLURM cannot create the log folder itself, and a job whose log
cannot be written fails without a message.

## 11. Step 0: checks

Submit all three; `gpu_test` and `cpu_short` each run one of your jobs at a time, so the
others wait their turn:

```bash
sbatch slurm/checks.sbatch
sbatch --job-name=gpu-check --partition=gpu_test --time=00:10:00 slurm/job.sbatch \
  python -c "import torch; x = torch.randn(256, 256, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'matmul ok', float((x @ x).sum()))"
sbatch slurm/net_check.sbatch
bash slurm/net_check.sbatch
```

The last line runs the network check on the login node, for comparison. When `squeue -u $USER`
no longer lists the jobs, read the logs (`ls -t runs/logs | head`, then `less FILE`; Space
pages, `q` quits):

| Log | Expected |
|---|---|
| `step0-checks-*.out`, `=== tests` | `N passed, M skipped`, no failure. The skips are datasets absent on Ruche (raw files) and Paderborn/PHM 2010 acceptance tests. The five `test_encoders.py` checkpoint tests must **pass**: that proves the weights load offline |
| `step0-checks-*.out`, `=== window stores` | The four stores, with the keys of the laptop |
| `gpu-check-*.out` | Ends with one line like `2.x+cu126 NVIDIA A100-SXM4-40GB (8, 0) matmul ok …` (or a V100 on `gpu_test`), with no warning above it |
| `net-check-*.out` | A `VERDICT` line: see below |

| Network verdict on the compute node | Meaning |
|---|---|
| `no internet on …` | Expected. Weights come from §8; data only from the login node |
| `all 9 hosts reachable` | Fine; jobs stay offline anyway, for reproducibility |
| `k of 9 reachable; failing: …` | A proxy or firewall; send me the log |

If `paderborn` also fails on the **login** node, Ruche cannot fetch Paderborn either: use the
VPN route in `docs/USER_ACTIONS.md` (step 2) and build the store on the laptop instead of §12.

## 12. Step 1: Paderborn

**12.1 Download, on the login node** (several GB; resumable; the SHA-256 of every archive is
recorded and verified). `nohup` keeps it running if your SSH session closes:

```bash
source slurm/env.sh
nohup bash slurm/fetch_paderborn.sh > runs/logs/paderborn-fetch.log 2>&1 &
tail -f runs/logs/paderborn-fetch.log
```

Ctrl+C stops the watching, not the download. If the session dies anyway, run the same line
again: finished files are kept and a partial one resumes. It ends with
`paderborn: 32 files verified, 0 added, 0 mismatched, 0 missing`.

**12.2 A RAR extractor.** The archives are RAR files; `pdmfm.fetch extract` uses `7z`, then
`unrar`:

```bash
which 7z 7za unrar
```

If none is found, install 7-Zip into the environment and check that it reads RAR:

```bash
conda install -n pdmfm -y -c conda-forge p7zip
source slurm/env.sh && 7z i | grep -i rar
```

The second command must print a line containing `Rar`. If it prints nothing, the build has no
RAR support. Use `module avail 7zip p7zip unrar` for a cluster module, or the laptop route at
the end of this section.

**12.3 Extract, test and build the window store** (one CPU job):

```bash
sbatch --job-name=paderborn slurm/cpu.sbatch bash -c "python -m pdmfm.fetch extract paderborn && python -m pytest -o addopts= -q -rs -m data -k paderborn tests/test_real_data.py && python -m pdmfm.windows build paderborn && python -m pdmfm.features texts --dataset paderborn"
```

The log `runs/logs/paderborn-*.out` must show:

1. the extraction;
2. `2 passed`: the acceptance tests, which check 32 bearings × 80 files, 64 kHz signals of
   about 4 s, and speeds that match the settings;
3. `[windows] paderborn: N windows x 3 channels`, with N about 7,680 (3 windows per 4 s file);
4. `[texts] paderborn: 4 contexts, 3 prototypes`: the context sentence of each operating
   setting, embedded now that the store exists (Qwen3 from the §8 cache, on the CPU).

Then on Ruche, in `pdmfm/datasets/paderborn.py`, set `VERIFIED_ON_REAL_DATA = True`. Commit
and push that change from the laptop, so both copies agree.

If the job runs out of its hour (`DUE TO TIME LIMIT`), submit it again: extraction skips
finished archives. Or give it a longer partition (`sinfo -s`).

**Laptop route** (no RAR tool on Ruche):

1. Copy `$PDMFM_DATA_ROOT/raw/paderborn` to `D:\Project14\data\raw\` with `scp -r`.
2. On the laptop, run `python -m pdmfm.fetch verify paderborn`, `extract`,
   `pytest -m data -k paderborn` and `python -m pdmfm.windows build paderborn`.
3. Upload `interim/windows/paderborn` as in §1 and §5.

## 13. Step 2: feature caches

Each command encodes every window of the listed datasets, once per arm, and resumes where it
stopped if a job ends early. A dataset without a window store (Paderborn before §12) is
skipped with a message, so you can start before Paderborn is ready and rerun the same command
afterwards: only the new windows are encoded.

**Handcrafted reference** (CPU, a few minutes):

```bash
sbatch --job-name=features-stats slurm/cpu.sbatch \
  python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda paderborn --encoder stats --arms T F E
```

**Mantis, MOMENT and AST** (one A100 each):

```bash
for encoder in mantis moment; do
  sbatch --job-name=features-$encoder --time=06:00:00 slurm/job.sbatch \
    python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda paderborn --encoder $encoder --arms T F E --device cuda
done
sbatch --job-name=features-ast --time=04:00:00 slurm/job.sbatch \
  python -m pdmfm.features extract --dataset cwru xjtu_sy paderborn --encoder ast --arms S --device cuda
```

- The E arm exists only for the bearing datasets, and the S arm only at 12 kHz. The commands
  skip those arms elsewhere, with a message (`no E arm; skipped`).
- Each log shows one progress line per chunk, with the encoding rate in windows per second.
- Mantis features exist on the laptop already. Encoding them again here takes minutes and
  keeps every compared encoder on one machine.
- **KAIST** (target, stage 8 only): the same commands with `--dataset kaist` encode it for
  later. Gate 1 never reads it.

`--time` is lower than the 24 h maximum, so the scheduler can start a job sooner. If a job
reaches its limit, submit the same command again: cached windows are skipped.

## 14. Step 3: own PatchTST, one model per outer fold

Each fold's model sees no target dataset and none of that fold's test or inner-validation
units (`pdmfm.protocols.pretraining_mask`). This loop pretrains the five folds, and for each
one queues a feature job that starts only if the pretraining succeeded:

```bash
for k in 1 2 3 4 5; do
  job=$(sbatch --parsable --job-name=patchtst-fold$k --time=06:00:00 slurm/job.sbatch \
    python -m pdmfm.patchtst --fold $k --device cuda)
  sbatch --dependency=afterok:${job%%;*} --job-name=patchtst-fold$k-features --time=04:00:00 slurm/job.sbatch \
    python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda paderborn \
    --encoder patchtst --checkpoint runs/patchtst/fold$k/model --arms T F E --device cuda
done
```

- Up to 4 of your A100 jobs run at once; the rest wait (`QOSMaxGRESPerUser` in `squeue`).
- A pretraining log shows one line per epoch:
  `[patchtst] epoch 3/20: train 0.4102, held-out 0.4321, 52 s`.
  The best model is in `runs/patchtst/fold<k>/model/`, the curve in `metrics.json`.
- **Run it after §12**, so the corpus includes Paderborn's vibration. Started earlier, the
  corpus lacks it: `metrics.json` lists the datasets used. Retrain with a new `--out` if you
  want it included.
- A pretraining job that reaches its time limit resumes from its last epoch when you submit
  the same `sbatch … python -m pdmfm.patchtst --fold k …` line again.

## 15. Step 4: Gate 1

When steps 1-3 are complete (`bash slurm/status.sh` shows every cache `True` under
`complete`, and five PatchTST runs `complete`):

```bash
sbatch --job-name=gate1 slurm/cpu.sbatch python -m pdmfm.probe --run-id gate1-v1 \
  --candidates mantis:T mantis:F mantis:E mantis:T+E moment:T moment:F moment:E moment:T+E \
  "patchtst-fold{fold}:T" "patchtst-fold{fold}:F" "patchtst-fold{fold}:E" "patchtst-fold{fold}:T+E" ast:S \
  --references stats:T stats:F stats:E
```

- Keep the quotes around the `patchtst-fold{fold}` candidates. `{fold}` makes fold k's task
  use fold k's own model.
- The probe fits the candidates of each split in parallel, one process per granted CPU
  (8 here). If the log ends with `DUE TO TIME LIMIT`, resubmit with a longer CPU partition
  from `sinfo -s` (`--partition=… --time=03:00:00` before `slurm/cpu.sbatch`), or run the
  same command on the laptop after §17.
- The result is in `runs/gate1/gate1-v1/`:
  - `summary.md`: the table and the winner;
  - `summary.csv`: scores and ranks per task;
  - `results.csv`: every fold, with the chosen regularization;
  - `report.json`: the winner and anything skipped.
- The rule, fixed before any result: the lowest mean rank over the four tasks wins; a tie goes
  to the cheaper encoder. A candidate without an applicable arm takes the worst rank on that
  task.
- If no frozen encoder beats your own PatchTST, the plan's Gate 1 picks the self-pretrained
  PatchTST.

## 16. Step 5: Plan B

Plan B is the plan's stage 4 and its fallback model: one evidential expert per modality,
combined by Dempster's rule for the classes and by MoNIG for wear and RUL. It trains on the
cached features, so it encodes nothing new. Each job runs the 5 outer folds × 3 seeds of one
dataset and ends with a summary table. The laptop already ran it with Mantis on Nonastreda
and XJTU-SY (`PROJECT_LOG.md`, week 5); here it runs with the Gate-1 winner, and on
Paderborn.

**16.1 Take the encoder and arms from the Gate-1 winner** (after §15):

```bash
WINNER=$(python -c "import json; print(json.load(open('runs/gate1/gate1-v1/report.json'))['winner'])")
ENCODER=${WINNER%%:*}
ARMS=$(echo "${WINNER#*:}" | tr '+' ' ')
echo "encoder $ENCODER, arms $ARMS"
```

The last line prints, for example, `encoder mantis, arms F` or `encoder patchtst-fold{fold},
arms T E`. Keep `{fold}` as it is: Plan B then uses fold k's own PatchTST on fold k, as Gate 1
did. Nonastreda has no E arm. If the winner's arms are E only, use `--arms F` for Nonastreda
(its best arm in Gate 1).

**16.2 Tune on the inner-validation units** (one GPU job per dataset; it never reads test
units):

```bash
for dataset in nonastreda xjtu_sy paderborn; do
  sbatch --job-name=tune-$dataset --time=04:00:00 slurm/job.sbatch \
    python -m pdmfm.tune --dataset $dataset --encoder "$ENCODER" --arms $ARMS --run-id tune-v1-$dataset --grid d=64,256 dropout=0.1,0.3 lr=3e-4,1e-3 --device cuda
done
```

Each writes `runs/tune/tune-v1-<dataset>/best.json`: the chosen settings, the ranking, and
the flags for Plan B. Near-ties (within one standard error of the paired fold-wise
differences) go to the smaller model. On the laptop, the same search chose the defaults for
Nonastreda and d = 64 for XJTU-SY (`PROJECT_LOG.md`, week 5).

**16.3 Submit Plan B** with the tuned flags (one GPU job per dataset), then the handcrafted
reference and a photo-free Nonastreda run:

```bash
for dataset in nonastreda xjtu_sy paderborn; do
  FLAGS=$(python -c "import json; print(json.load(open('runs/tune/tune-v1-$dataset/best.json'))['flags'])")
  echo "$dataset: $FLAGS"
  sbatch --job-name=planb-$dataset --time=02:00:00 slurm/job.sbatch \
    python -m pdmfm.planb --dataset $dataset --encoder "$ENCODER" --arms $ARMS --run-id planb-v1-$dataset $FLAGS --device cuda
  sbatch --job-name=planb-$dataset-stats --time=02:00:00 slurm/job.sbatch \
    python -m pdmfm.planb --dataset $dataset --encoder stats --arms F E --run-id planb-v1-$dataset-stats --device cuda
done
sbatch --job-name=planb-nonastreda-nophoto --time=02:00:00 slurm/job.sbatch \
  python -m pdmfm.planb --dataset nonastreda --encoder "$ENCODER" --arms $ARMS --run-id planb-v1-nonastreda-nophoto --no-photos --device cuda
```

- Keep the quotes around `"$ENCODER"`: they protect `{fold}`.
- Level tokens are relative to each unit's healthy baseline (`--levels relative`, the
  default and the plan's form). `--levels both` (absolute descriptors) and `--force-mean`
  (the static cutting force) are ablations, each run with a new run id
  (`PROJECT_LOG.md`, week 5).
- The handcrafted reference needs the §13 `stats` caches. Arms a dataset lacks are skipped,
  so `--arms F E` is right for all three.
- Up to 4 of your A100 jobs run at once; the rest wait.
- A log shows the inputs once, then two lines per fold and seed (`<…>` are run values):

  ```text
  [planb] nonastreda mantis F: modalities ['force', 'photo']; <settings key>
  [planb] fold 1 seed 0: train 3046, validation 428, test 767 windows
  [train] 50 epochs in <seconds> s; best epoch <epoch> (<validation score>)
  ```

  The laptop CPU took about 70 s per fold and seed on Nonastreda; an A100 should take
  seconds. The log ends with `[planb] done in … s -> …/runs/planb/planb-v1-nonastreda`. Paderborn prints
  the `modalities` line once per fold: its level baseline is the healthy *training*
  bearings of each fold, so its inputs differ by fold.
- `bash slurm/status.sh` shows each run under `== Plan B runs` as `15 of 15` when complete.
- The table is in `runs/planb/<run-id>/summary.md`: test units, mean ± std over folds and
  seeds, before and after temperature scaling, each modality alone, the model without each
  modality, and the majority, persistence and training-mean baselines. Every fold and seed
  is in `fold<k>-seed<s>.json`, with the test predictions in `.npz` beside it.

**16.4 Check Paderborn's temperature.** In `runs/planb/planb-v1-paderborn/summary.md`, look
at the `only_slow` row: the temperature alone. On KAIST, temperature followed the recording
session, not the fault (EDA finding 2). If temperature alone reaches a macro-F1 near the
full model's, it probably reflects the session here too. Then run Paderborn again without it:

```bash
sbatch --job-name=planb-paderborn-noslow --time=02:00:00 slurm/job.sbatch \
  python -m pdmfm.planb --dataset paderborn --encoder "$ENCODER" --arms $ARMS --run-id planb-v1-paderborn-noslow --without slow --device cuda
```

Report both tables; I decide with you which one stands.

**Resuming.** A job that stops at its time limit continues when you submit the same command
again: finished folds and seeds are skipped. The same `--run-id` with other settings is
refused (`… holds a run with other settings; use another --run-id`), so a changed option
needs a new run id.

## 17. Step 6: download

Pack on Ruche (small; the login node is fine):

```bash
cd "$WORKDIR/pdmfm" && tar -czf gate1-results.tar.gz runs/gate1 runs/patchtst runs/tune runs/planb runs/logs
cd "$WORKDIR/pdmfm_data" && tar -czf features.tar.gz interim/features
tar -czf paderborn-windows.tar.gz interim/windows/paderborn
ls -lh "$WORKDIR/pdmfm/gate1-results.tar.gz" features.tar.gz paderborn-windows.tar.gz
```

Then on the laptop:

```powershell
cd "C:\Users\ali_a\OneDrive\Desktop\Multimodal AI for Predictive Maintenance"
scp "${Ruche}:/workdir/$RucheLogin/pdmfm/gate1-results.tar.gz" .
tar -xzf gate1-results.tar.gz
cd D:\Project14\data
scp "${Ruche}:/workdir/$RucheLogin/pdmfm_data/features.tar.gz" .
scp "${Ruche}:/workdir/$RucheLogin/pdmfm_data/paderborn-windows.tar.gz" .
tar -xzf features.tar.gz
tar -xzf paderborn-windows.tar.gz
```

`runs/` is git-ignored, so the results stay out of commits. Ruche's directories are not backed
up: keep the archives on the laptop.

---

# Part C: Operating

## 18. Watch, resume and cancel

| Task | Command |
|---|---|
| Everything at a glance | `bash slurm/status.sh` |
| Your jobs | `squeue -u $USER` (`PD` waiting, `R` running; the last column is the node or the reason) |
| Follow a log | `tail -f runs/logs/NAME-JOBID.out` (Ctrl+C stops watching, not the job) |
| A finished job's state and resources | `sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode`, `seff JOBID` |
| Cancel | `scancel JOBID`; all your jobs: `scancel -u $USER` |
| Cache progress | `python -m pdmfm.features status` |

**Every step resumes.** Feature extraction saves after each chunk, PatchTST saves after
each epoch, and Plan B saves after each fold and seed. A job stopped at its limit (`TIMEOUT`, `DUE TO TIME LIMIT` at the end of the log)
continues when you submit the same command again. For work longer than 24 h, `bash
slurm/chain.sh PARTS [options] slurm/job.sbatch COMMAND` submits consecutive parts.

**Never run two jobs on the same cache or run folder at the same time.** One command per
encoder, as in §13, never collides.

## 19. Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| The job ends at once and no log exists | `runs/logs` missing at submission | `mkdir -p runs/logs` (§10), resubmit |
| `sbatch: error: invalid partition specified` | Partition names changed | `sinfo -s`; use the A100 (or CPU) partition it lists |
| `module: command not found` or an Lmod error | Wrong module in `slurm/env.sh` | `module avail anaconda miniconda`; correct `CONDA_MODULE` (§7) |
| `ModuleNotFoundError: No module named 'pdmfm'` (or `momentfm`, `mantis`) | Wrong `CONDA_ENV`, or §6 unfinished | `conda env list`; repeat §6 |
| `LocalEntryNotFoundError`, `couldn't connect to 'https://huggingface.co'`, or offline-mode errors | A checkpoint missing from `$HF_HOME` | `source slurm/env.sh && bash slurm/prefetch_weights.sh` on the login node |
| `CUDA was requested but is not available`, or a warning that PyTorch "does not include kernels for this GPU" | PyTorch build and GPU or driver mismatch | `source slurm/env.sh`, `pip uninstall -y torch`, then reinstall from `…/whl/cu126` (§6) and repeat the GPU check |
| `CUDA out of memory` | Batch too large for the GPU | Add `--batch-size 32` (features) or `--batch-size 256` (PatchTST: new `--out`, because the settings change) |
| `no window store in …` | Store not uploaded or not built | §5, or §12 for Paderborn |
| `… holds other features; delete it to rebuild` | A cache folder from other settings | Delete that folder under `$PDMFM_DATA_ROOT/interim/features/<dataset>/` |
| `several … caches for …` (Gate 1) | Two caches of one encoder and arm, e.g. after a code update | Keep the newest folder (`ls -lt`) and delete the other |
| `no <encoder> <arm> features for <dataset>` (Plan B) | The cache of that encoder, arm and dataset is missing | §13 (or §14 for PatchTST); `python -m pdmfm.features status` lists what exists |
| `<dataset>: no <encoder> features for arms ('E',)` (Plan B, Nonastreda) | Nonastreda has no E arm | `--arms F` for Nonastreda (§16.1) |
| `… holds a run with other settings; use another --run-id` (Plan B) | Same run id, other options or caches | A new `--run-id` |
| `no modality ['…']; <dataset> has […]` (Plan B) | `--without` names a modality the dataset lacks | Use a name from that list |
| `… was started with other settings or data; use a new --out` (PatchTST) | Resubmitted with other options, or the corpus changed (Paderborn added) | The original command, or `--out runs/patchtst/fold<k>-v2`. The encoder is then named `patchtst-fold<k>-v2`: extract its features with `--checkpoint runs/patchtst/fold<k>-v2/model` and use the candidate `"patchtst-fold{fold}-v2:T"` |
| `could not extract K001.rar` | No RAR tool | §12.2 |
| `Disk quota exceeded` | Home or workdir full | `ruche-quota`; move `~/.conda` (§3); `rm -rf ~/.cache/pip` |
| `git clone` asks for a password and refuses it | GitHub needs a token, not your password | §4 |

**Interactive debugging on a GPU** (at most 30 minutes):

```bash
srun --partition=gpu_test --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash
cd "$WORKDIR/pdmfm" && source slurm/env.sh && nvidia-smi
```

## 20. Rules of the cluster

- **No computing on the login nodes.** Installing, downloading, moving files, submitting jobs
  and `slurm/status.sh` are fine; tests, encoding, training and the probe go into jobs.
- Request what you use: one GPU and 8 CPUs per job (the defaults).
- Prefer the A100s (`gpua100`).
- Never edit `~/.bashrc` and never run `conda init`.
- Back up results yourself (§17).
- Help: <ruche.support@universite-paris-saclay.fr>. Give your login, lab, job ID, the problem
  and the log.

## 21. Cheat sheet

After the §10 block, from `$WORKDIR/pdmfm`:

| Step | Command |
|---|---|
| Status | `bash slurm/status.sh` |
| 0 | `sbatch slurm/checks.sbatch` · GPU check (§11) · `sbatch slurm/net_check.sbatch` |
| 1 | `nohup bash slurm/fetch_paderborn.sh > runs/logs/paderborn-fetch.log 2>&1 &` (login), then the §12.3 job |
| 2 | `sbatch --job-name=features-stats slurm/cpu.sbatch python -m pdmfm.features extract --dataset cwru xjtu_sy nonastreda paderborn --encoder stats --arms T F E` and the §13 loop |
| 3 | The §14 loop |
| 4 | The §15 command |
| 5 | §16.1 (winner), the §16.2 tuning loop, then the §16.3 Plan B loop |
| 6 | §17 |
