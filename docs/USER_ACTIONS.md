# Steps only you can take

Everything here needs your accounts, your credentials, a cluster session or a decision. The
code, checks and tests behind each step are ready, and each step says what success looks
like. Commands for the laptop assume the development environment is loaded first:

```powershell
. .\scripts\env\dev.ps1      # PDMFM_DATA_ROOT=D:\Project14\data, activates D:\Project14\.venv
```

| # | Step | Unblocks | Effort |
|---|---|---|---|
| 1 | Commit and push the Gate-2 stage (on top of `a10b2b7`); check that CI is green | Ruche clones the repository from GitHub | 5 min |
| 2 | Run the Ruche runbook (`docs/RUCHE_RUNBOOK.md`). The §1 archive now also carries the MATWI window store; step 5 (§16) now ends with Gate 2 (§16.6) and the RUL runs (§16.7) | MOMENT, AST and PatchTST caches; Paderborn; the full Gate 1; Plan B, Gate 2 and RUL with the Gate-1 winner | about 3 h of your time over a few days, the rest is queue and compute |
| 3 | Download PHM 2010 from Kaggle (needs your Kaggle account; I cannot log in for you) | Milling wear-transfer target (Goal 2); milling physics score | 15-30 min |
| — | Decisions A-E taken; F (second paper) and G (C-MAPSS) deferred, as you asked (section 4) | — | — |
| — | Done: GitHub repository, CI, pre-commit hook; commits `1a53d64`, `b837e80`, `965d479`, `84f4add`, `a10b2b7`; MATWI downloaded (18 GB, every file checked against the publisher's MD5) and built | — | — |

---

## 1. Commit and push the Gate-2 stage

`a10b2b7` holds tuning and the first MATWI pieces. Not committed yet: the MATWI reader and
store support, the Gate-2 candidates and search, the RUL model, the OmniFuser lessons
(schedule baseline, ordinal metrics), their tests and the docs.

```powershell
cd "C:\Users\ali_a\OneDrive\Desktop\Multimodal AI for Predictive Maintenance"
git status -sb   # new: pdmfm/datasets/matwi.py, pdmfm/gate2.py, pdmfm/model/fusion.py, pdmfm/rul.py, pdmfm/texts/matwi.yaml, 3 test files
git add -A
git commit -m "MATWI, Gate-2 fusion search with inner cross-validation, RUL from past snapshots"
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
   tune each dataset on its validation units (§16.2), then submit the §16.3 loop: Nonastreda,
   MATWI, XJTU-SY and Paderborn with the winner and with the handcrafted reference, plus a
   photo-free Nonastreda run. Check Paderborn's temperature as §16.4 says. Then the Gate-2
   loop (§16.6: Nonastreda, MATWI, Paderborn; a few hours per job) and its test-unit run, and
   the RUL loop (§16.7, CPU jobs).
7. **Step 6 (§17):** download the caches and results (now with `runs/planb`, `runs/gate2`
   and `runs/rul`), then tell me. I write up Gate 1, Plan B, Gate 2 and RUL from them.

If you run only part of the runbook before Gate 1 is final, Plan B can still run on Paderborn
with Mantis (its caches come from §13): use `--encoder mantis --arms F` in the §16.2 and §16.3 loops and
a run id such as `planb-mantis-paderborn`.

---

## 3. PHM 2010 milling data (Kaggle)

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

**Place and check.**

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

| Week | Stage | What I build and test locally | What then needs Ruche |
|---|---|---|---|
| 6-7 (Nov 30 - Dec 13) | **Gate 2: fusion search** | The four other fusion candidates on the cached tokens, each ≤ 3 M parameters, sharing Plan B's inputs, heads, training loop and evaluation: concat-MLP, Perceiver latents (32 latents, 2 cross- + 4 self-attention layers), MBT bottleneck (4 tokens, 4 layers), C2F-N (sensor hub with gated landmarks per other modality). AdamW 3e-4 with a cosine schedule, 50 epochs, modality dropout 0.3. Then missing-modality curves and the Gate-2 rule (validation macro-F1 or RMSE, then robustness, then ECE; ties to the smaller model). Runs locally on Nonastreda and XJTU-SY | The same search with the Gate-1 winner and on Paderborn: one runbook section, like §16 |
| 8-9 (Dec 14-27) | Fusion pretraining | Masked latent modelling on cached tokens, cross-modal InfoNCE, text anchoring; corpus minus each fold's held-out units; smoke tests on CPU | The pretraining jobs |
| 10-11 | Adaptation | Few-shot draws, linear probe vs LoRA vs full fine-tuning on KAIST, Paderborn real damage, PHM 2010 | The sweeps (holiday queue) |

If Gate 2 finds no candidate better than late aggregation on the validation units, Plan B
becomes the model (plan, Gate 2), and weeks 8-9 pretrain its per-modality adapters instead.

Carried into Gate 2 from the audit and the tests of 2026-10-30 (`PROJECT_LOG.md`, week 5):
- **Inner cross-validation for every option choice.** One validation tool per fold is too
  noisy to choose options on Nonastreda (every option looked better on test units, none on
  validation). Rotating the validation tool over the training tools needs about 7 times the
  fits: a Ruche job.
- **Temporal context for RUL.** A single snapshot cannot place a bearing on its degradation
  path: after the first prediction time the error is 0.29 and the PHM 2012 score 0.20. Next: a
  causal model over the past snapshots' embeddings, then the plan's TimesFM stage.
- **The photo options** (older photos, augmented copies, modality dropout) re-tested under
  inner cross-validation; modality dropout is part of every Gate-2 candidate anyway.
- **A steadier interval calibration.** Fit decision D's scale on cross-fitted predictions of
  the training units (7-9 units instead of 1-3), because the current scale worsened XJTU-SY
  coverage.
- **The supervised from-scratch PatchTST baseline** that the plan lists for every table.
- **Paired tests over fold × seed** (Wilcoxon, bootstrap confidence intervals, Holm
  correction) in the summaries.
- **The text projection W_t before any zero-shot claim.** Class prototypes are nearly
  parallel in Qwen space.

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
