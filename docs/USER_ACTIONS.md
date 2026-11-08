# Steps only you can take

Everything here needs your accounts, your credentials, a cluster session or a decision. The
code, checks and tests behind each step are ready, and each step says what success looks
like. Commands for the laptop assume the development environment is loaded first:

```powershell
. .\scripts\env\dev.ps1      # PDMFM_DATA_ROOT=D:\Project14\data, activates D:\Project14\.venv
```

| # | Step | Unblocks | Effort |
|---|---|---|---|
| 1 | Commit and push the week-8 write-up (docs only, on top of `c4f4ac0`; `a7716ec` and `c4f4ac0` hold the code); check that CI is green for `c4f4ac0` | Ruche clones the repository from GitHub | 5 min |
| 2 | Run the Ruche runbook (`docs/RUCHE_RUNBOOK.md`), following its **run-order table** (steps 0-8): checks, Paderborn, caches, PatchTST, Gate 1, nested tuning, Plan B, Gate 2, RUL, baselines, then pretraining (§16.9) and fine-tuning from it (§16.10), then the download | Everything from Gate 1 to stage 7 with the winning encoder | about 3-4 h of your time over a few days, the rest is queue and compute |
| 3 | Download PHM 2010 from Kaggle, on Ruche: runbook §12.4, step 1b of its run order (needs your Kaggle account and API token; I cannot log in for you) | Milling wear-transfer target (Goal 2); milling physics score | 15-30 min |
| — | Decisions A-E taken; F (second paper) and G (C-MAPSS) deferred, as you asked (section 4) | — | — |
| — | Done: GitHub repository, CI, pre-commit hook; commits `1a53d64`, `b837e80`, `965d479`, `84f4add`, `a10b2b7`, `b2b7d4d` (MATWI, Gate 2, RUL), `6866791` (week-6 docs), `3103e5e` and `e269b1d` (week 7), `07638c0` (channel descriptions, nested selection), `a7716ec` (pretraining), `c4f4ac0` (fine-tuning takes the backbone's width); MATWI downloaded (18 GB, every file checked against the publisher's MD5) and built | — | — |

---

## 1. Commit and push the week-8 write-up

`a7716ec` holds this week's code: `pdmfm/pretrain.py` (stage 6), fine-tuning from it
(`planb --init`, stage 7), the model hooks they need, `tests/test_pretrain.py`, the runbook's
run-order table and §16.9-§16.10, and the status script's pretraining section. `c4f4ac0`
adds one fix: fine-tuning takes the backbone's width and head (Ruche tunes `d` over {64, 256}
from scratch; one backbone of width 256 serves every dataset). Still to commit: the week-8
results and the progress review in `PROJECT_LOG.md`, `docs/CI.md` and this file.

```powershell
cd "C:\Users\ali_a\OneDrive\Desktop\Multimodal AI for Predictive Maintenance"
git status -sb   # PROJECT_LOG.md, docs/CI.md, docs/USER_ACTIONS.md
git add -A
git commit -m "Week 8: pretraining pilot and progress review"
git push
```

`runs/` is git-ignored, so the local results stay out of the commit. Then check that CI is
green on GitHub (lint, 4 test legs, advisory pyright). If a job is red, paste its log.

---

## 2. Ruche

Follow `docs/RUCHE_RUNBOOK.md` from the start. It has every command and what each log should
show. In short:

1. **Set up once (§1-§9):**
   - pack and upload the four window stores;
   - clone the repository;
   - create the `pdmfm` conda environment;
   - set `slurm/env.sh`;
   - download the weights on the login node.
2. **Step 0 (§11):** checks, GPU check and network check. Read the `VERDICT` line as the
   runbook explains.
3. **Step 1 (§12): Paderborn.**
   - Download on the login node.
   - Extract, run the acceptance tests and build the window store in a CPU job.
   - Then set `VERIFIED_ON_REAL_DATA = True` in `pdmfm/datasets/paderborn.py`.
   - If Ruche cannot reach the Paderborn host either, connect through a university VPN and run
     `python -m pdmfm.fetch get paderborn --workers 4` on the laptop. Build the store there
     (runbook §12, "Laptop route") and upload it.
   - Licence CC BY-NC 4.0: research use is fine; do not redistribute the files.
4. **Steps 2-3 (§13-§14):** the feature jobs, and the five PatchTST pretraining jobs with
   their feature jobs.
5. **Step 4 (§15):** the Gate-1 probe on all four tasks.
6. **Step 5 (§16): Plan B, Gate 2 and RUL.** Read the Gate-1 winner with the §16.1 lines,
   tune each dataset by nested cross-validation (§16.2; up to 8 h per job), then submit the
   §16.3 loop, which scores each fold with its own tuned settings (`--per-fold`): Nonastreda,
   MATWI, XJTU-SY and Paderborn with the winner and with the handcrafted reference, plus a
   photo-free Nonastreda run. Check Paderborn's temperature as §16.4 says. Then the Gate-2
   loop (§16.6: Nonastreda, MATWI, Paderborn; a few hours per job) and its test-unit run, and
   the RUL loop (§16.7, CPU jobs). Then §16.8: the baselines (one CPU job for gradient
   boosting and the probes, one PatchTST GPU job per dataset) and the physics check of the
   Gate-1 winner (one GPU job); they do not wait for Gate 2. Then §16.9: pretraining on all
   five source datasets (one job per fold, plus the three loss ablations) and §16.10:
   Plan B fine-tuned from each fold's own backbone.
7. **Step 6 (§17):** download the caches and results (now with `runs/planb`, `runs/gate2`,
   `runs/rul`, `runs/baselines`, `runs/explain` and `runs/pretrain`), then tell me. I write up Gate 1, Plan B,
   Gate 2 and RUL from them, with the paired reports, the leakage checks and the anomaly
   tables, which run on the laptop from the downloaded caches.

If you run only part of the runbook before Gate 1 is final, Plan B can still run on Paderborn
with Mantis (its caches come from §13): use `--encoder mantis --arms F` in the §16.2 and §16.3 loops and
a run id such as `planb-mantis-paderborn`.

---

## 3. PHM 2010 milling data (Kaggle)

**Recommended: on Ruche**, with the Kaggle command-line tool and your API token:
`docs/RUCHE_RUNBOOK.md` §12.4 has every command (token, download, verify, the extraction
and acceptance-test job, the window store, clean-up), and §13 the target caches. The store
comes back to the laptop with the §17 download, so the laptop route below is needed only
if Ruche cannot reach Kaggle.

**Why.** The plan's second milling target (wear regression, leave-one-cutter-out), and the
only milling set with a known spindle speed, which the physics score needs. The reader
(`pdmfm/datasets/phm2010.py`) is written against the challenge description but has never
seen the real files (`VERIFIED_ON_REAL_DATA = False`).

**Where.** I checked every route on 2026-10-31; none can be fetched without your login:

- the original host (`cdn.teamholistic.net`) serves a parking page, and the PHM Society
  data repository links only that dead page;
- IEEE DataPort lists the challenge without files (subscription);
- **Kaggle** holds a copy of the challenge,
  [`rabahba/phm-data-challenge-2010`](https://www.kaggle.com/rabahba/phm-data-challenge-2010):
  six cutters c1-c6, about 300 per-cut CSV files each, and a wear file for c1, c4 and c6
  (flank wear after each cut, in 10⁻³ mm). Downloading needs a Kaggle account. Its page warns
  that one `c3.zip` copy with a wear file is incorrect: c3 is unlabelled in the challenge, and
  we use c2, c3 and c5 only for pretraining, so ignore any c3 wear file.

Take the original per-cutter zips (`c1.zip` … `c6.zip`), not a repackaged version.

Do not use the Hugging Face mirror `AI4Manufacturing/PHM2010`. It is gated, states no
licence and has been converted to parquet.

**Place and check (laptop route).** Download with the browser (the dataset page's
**Download** button), unzip that archive, and take the `c*.zip` files inside it:

```powershell
# c1.zip, c4.zip, c6.zip are required (labelled); c2, c3, c5 are optional (pretraining only)
Copy-Item <where you saved them>\c*.zip D:\Project14\data\raw\phm2010\
python -m pdmfm.fetch verify phm2010     # records SHA-256 of hand-placed files in MANIFEST.json
python -m pdmfm.fetch extract phm2010
python -m pytest -m data -k phm2010 -rs  # acceptance tests
python -m pdmfm.windows build phm2010    # the window store (stage 1)
```

**Success.**
- `verify` reports `0 missing, 0 mismatched`.
- Both acceptance tests pass. They check:
  - 315 cuts for each of c1, c4 and c6, each with a wear value in µm that grows from cut to cut;
  - 7 columns per cut file;
  - a dominant force line at a multiple of 173.3 Hz, which confirms the commonly quoted 10,400
    rpm.

  Then set `VERIFIED_ON_REAL_DATA = True` in `pdmfm/datasets/phm2010.py` and tell me.

**If a test fails.**
- **Cut count or wear check fails:** the mirror packages the data differently. Send me the
  file listing (`Get-ChildItem -Recurse D:\Project14\data\extracted\phm2010 | Select -First 30`).
- **Spindle check fails:** its message names the measured line. `SPINDLE_RPM` is then wrong,
  and I correct it before the physics score uses it.

**If you cannot obtain it:** NASA milling is already downloaded and verified, but it is a weak
substitute (250 Hz, RMS-processed signals, 16 cases). See decision A.

---

## 4. Decisions

| | Decision | What follows from it |
|---|---|---|
| A | **Milling transfer target:** PHM 2010 if step 3 succeeds; otherwise NASA milling, with the report stating its limits | Stage 8 uses PHM 2010; if it cannot be obtained, NASA milling (250 Hz, RMS-processed signals, 16 cases) with its limits written into the report |
| B | **Paderborn through Ruche** | Runbook §12; the VPN route is only the fallback if Ruche cannot reach the host |
| C | **CWRU for pretraining and Gate 1 only**, never as evidence | No CWRU row in any result table; it stays one of Gate 1's four tasks and in the PatchTST corpus |
| D | **Calibrated wear and RUL intervals** | Implemented: one scale factor on the predictive interval, fitted by NLL on the inner-validation units; every regression table shows `combined_calibrated` next to `combined` |
| E | **MATWI added** (2026-10-31) | Downloaded in full (18 GB fit on D:), reader, insert crops, window store (11,043 windows), photo tokens and text prototypes built; a source dataset with 5 outer folds |

### Deferred (2026-10-31, at your request)

| | Decision | Status |
|---|---|---|
| F | **A second, short paper** on shortcuts in public predictive-maintenance data | Deferred. The material keeps growing: OmniFuser's run-number shortcut, Nonastreda's visual labels that disagree with its own wear on 18 passes, and the schedule baselines now in every table |
| G | **C-MAPSS** for RUL | Deferred. The RUL model now reads past snapshots; revisit only if RUL becomes a headline result |

Kept dropped: battery data (outside the scope of rotating machinery and milling), LLM
fine-tuning (no instruction data; it would conflict with trustworthy output), and the KIT CNC
set (no photos and no wear labels).

---

## What comes next (my side)

Where things stand: stages 0-7 are written and tested; stage 6 (pretraining) was built and
piloted on the laptop this week (`PROJECT_LOG.md`, week 8, with the full progress review:
what exists, what is missing, the priorities and how each goal is evaluated). Results from
Gate 1 onward wait on your Ruche runs.

| Order | Work | Needs | Effort |
|---|---|---|---|
| 1 | Stage 8, adaptation (`pdmfm/adapt.py`): zero-shot through the fault descriptions, linear probe, LoRA, full fine-tuning; recipe chosen on Paderborn's artificial-damage validation bearings | Nothing to write; the backbones and Paderborn to run | 6-8 days |
| 2 | Stage 7 completion: Nonastreda's leave-one-tool-out from a backbone pretrained without Nonastreda | - | about a day |
| 3 | Stage 11 on the final model: keep the fine-tuned models, band occlusion through Plan B's inputs, explanation cards | - | 3-5 days |
| 4 | Your Ruche results: Gate 1, Gate 2, pretraining and fine-tuning reports | Step 2 | about a day |
| 5 | Improvements 1-2 (defect-line tokens, baseline-relative embeddings), ideally before the final pretraining; the evaluation fixes 4, 5, 7; E1 | Your go-ahead | days |
| 6 | Stage 10 completion (masked-modelling anomaly score, text score; TimesFM) | - | 2-3 days |
| 7 | Report, figures, release; then writing | All results | 3-4 days, then 2 weeks |

Improvements found this week, for your go-ahead (details and costs in `PROJECT_LOG.md`, week 7):

1. **Defect-line tokens** in the vibration expert (the line prominences the gradient-boosting
   baseline already uses): physics-grounded inputs that transfer across bearings; about a day.
2. **Baseline-relative embeddings** (each unit's healthy-baseline embedding subtracted):
   frozen features identify the machine perfectly; relative levels already halve it.
3. **Predict the wear change since the previous pass** on milling, where persistence is the
   strongest baseline; one or two days.
4. **One stratified test across datasets** for the paper's headline claims; hours.
5. **A label-free onset metric and an adaptive healthy reference** for XJTU-SY, whose FPT
   labels are an RMS rule; hours.
6. **Less weight on CWRU as evidence** (saturated, and solved without its physics).
7. **Healthy support from other recordings** in the stage-8 anomaly runs (KAIST's 4 Nm
   healthy data is one recording, so same-recording support flatters the false alarms).

Queued experiments (protocols fixed in `PROJECT_LOG.md`, week 7, "Queued experiments"):

- **E1. Task-specific adapters** (your question, 2026-11-03): single-task against joint Plan B
  on Nonastreda and MATWI; adapters or loss weights only if the tasks interfere, adopted by
  the Gate-2 rule; the class derived from the wear prediction reported in every case.
- **E2. Sensor-type and sampling-rate embeddings: done (2026-11-05)**, with token positions;
  validated in `PROJECT_LOG.md`, "Week 7, continued".

Carried forward:
- **Cross-tool generalization is the limit on both milling sets** (training macro-F1 0.93
  against 0.41-0.78 on test tools).
- **RUL:** Plan B's expert embedding, cross-fitted, as the input of the sequence model; the
  plan's health indicator with TimesFM; C-MAPSS only if you reopen decision G.
- **Nonastreda's labels:** 18 passes disagree with their own wear; T6 is the floor of every
  candidate.
- **A steadier interval calibration** (cross-fitted on the training units).
- **The text projection W_t** before any zero-shot claim.

---

## Done

- **GitHub repository:** `origin` is
  `github.com/taanakoum/Multimodal-AI-for-Predictive-Maintenance`, branch `master`.
- **CI:** it runs on pushes to `master`, on pull requests, or by hand (`docs/CI.md`).
- **pre-commit hook:** installed (2026-10-26) and runs on every commit.
  - It runs from any terminal, because it records `D:\Project14\.venv\Scripts\python.exe`.
  - A bare `pre-commit` works only after `. .\scripts\env\dev.ps1`; otherwise call
    `D:\Project14\.venv\Scripts\pre-commit.exe`.
- **VS Code (recommended):** **Ctrl+Shift+P → Python: Select Interpreter →
  `D:\Project14\.venv\Scripts\python.exe`**. New terminals then activate the environment, and
  the editor stops reporting the project's packages as missing.
