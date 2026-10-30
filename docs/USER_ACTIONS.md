# Steps only you can take

Everything here needs your accounts, your credentials, a cluster session or a decision. The
code, checks and tests behind each step are ready, and each step says what success looks
like. Commands for the laptop assume the development environment is loaded first:

```powershell
. .\scripts\env\dev.ps1      # PDMFM_DATA_ROOT=D:\Project14\data, activates D:\Project14\.venv
```

| # | Step | Unblocks | Effort |
|---|---|---|---|
| 1 | Commit and push the audit pass (on top of `965d479`); check that CI is green | Ruche clones the repository from GitHub | 5 min |
| 2 | Run the Ruche runbook (`docs/RUCHE_RUNBOOK.md`), now with step 5: Plan B (§16). The §1 archive now also carries the photo tokens and text prototypes | MOMENT, AST and PatchTST caches; Paderborn; the full Gate 1; Plan B with the Gate-1 winner and on Paderborn | about 2-3 h of your time over a few days, the rest is queue and compute |
| 3 | Obtain PHM 2010 (manual download) | Milling wear-transfer target (Goal 2); milling physics score | 15-30 min |
| — | Decisions A-D taken (section 4) | — | — |
| — | Done: GitHub repository, CI, pre-commit hook; commits `1a53d64`, `b837e80`, `965d479` | — | — |

---

## 1. Commit and push the audit pass

`965d479` ("Per Modality Expert Implementation") holds Plan B. This pass (decision D, the audit
fixes, the cleanup and the docs) is not committed:

```powershell
cd "C:\Users\ali_a\OneDrive\Desktop\Multimodal AI for Predictive Maintenance"
git status -sb                  # modified modules, tests and docs; scripts/week1/ deleted
git add -A                      # also stages the deletion of scripts/week1/ (superseded; PROJECT_LOG)
git commit -m "Audit: interval calibration, metric and baseline fixes, cleanup"
git push
```

`runs/` is git-ignored, so the local Plan B results stay out of the commit. The pre-commit
hook runs ruff. Then open the repository's **Actions** tab and check that CI is green (lint,
4 test legs, advisory pyright). If a job is red, paste its log.

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
6. **Step 5 (§16): Plan B.** Read the Gate-1 winner with the §16.1 lines. Then submit the
   §16.2 loop: Nonastreda, XJTU-SY and Paderborn with the winner and with the handcrafted
   reference, plus a photo-free Nonastreda run. Check Paderborn's temperature as §16.3 says.
7. **Step 6 (§17):** download the caches and results (now with `runs/planb`), then tell me. I
   write up Gate 1 from `runs/gate1/gate1-v1/` and the Plan B tables from `runs/planb/`.

If you run only part of the runbook before Gate 1 is final, Plan B can still run on Paderborn
with Mantis (its caches come from §13): use `--encoder mantis --arms F` in the §16.2 loop and
a run id such as `planb-mantis-paderborn`.

---

## 3. PHM 2010 milling data

**Why.** The plan's second milling target (wear regression, leave-one-cutter-out), and the
only milling set with a known spindle speed, which the physics score needs. The reader
(`pdmfm/datasets/phm2010.py`) is written against the challenge description but has never
seen the real files (`VERIFIED_ON_REAL_DATA = False`).

**Where.** The original host (`cdn.teamholistic.net`) now serves a parking page. Options:

1. The [PHM Society challenge page](https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/): check whether it links the files again.
2. IEEE DataPort (search "PHM 2010 data challenge"; free account).
3. A Kaggle mirror. Use it only if it holds the **original zips** (`c1.zip` … `c6.zip`) with
   per-cut CSV files and a `cN_wear.csv`, not a repackaged version.

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

## 4. Decisions (taken 2026-10-29)

| | Decision | What follows from it |
|---|---|---|
| A | **Milling transfer target:** PHM 2010 if step 3 succeeds; otherwise NASA milling, with the report stating its limits | Stage 8 uses PHM 2010; if it cannot be obtained, NASA milling (250 Hz, RMS-processed signals, 16 cases) with its limits written into the report |
| B | **Paderborn through Ruche** | Runbook §12; the VPN route is only the fallback if Ruche cannot reach the host |
| C | **CWRU for pretraining and Gate 1 only**, never as evidence | No CWRU row in any result table; it stays one of Gate 1's four tasks and in the PatchTST corpus |
| D | **Calibrated wear and RUL intervals** | Implemented: one scale factor on the predictive interval, fitted by NLL on the inner-validation units; every regression table shows `combined_calibrated` next to `combined` |

No decision is open.

---

## What comes next (my side)

| Week | Stage | What I build and test locally | What then needs Ruche |
|---|---|---|---|
| 6-7 (Nov 30 - Dec 13) | **Gate 2: fusion search** | The four other fusion candidates on the cached tokens, each ≤ 3 M parameters, sharing Plan B's inputs, heads, training loop and evaluation: concat-MLP, Perceiver latents (32 latents, 2 cross- + 4 self-attention layers), MBT bottleneck (4 tokens, 4 layers), C2F-N (sensor hub with gated landmarks per other modality). AdamW 3e-4 with a cosine schedule, 50 epochs, modality dropout 0.3. Then missing-modality curves and the Gate-2 rule (validation macro-F1 or RMSE, then robustness, then ECE; ties to the smaller model). Runs locally on Nonastreda and XJTU-SY | The same search with the Gate-1 winner and on Paderborn: one runbook section, like §16 |
| 8-9 (Dec 14-27) | Fusion pretraining | Masked latent modelling on cached tokens, cross-modal InfoNCE, text anchoring; corpus minus each fold's held-out units; smoke tests on CPU | The pretraining jobs |
| 10-11 | Adaptation | Few-shot draws, linear probe vs LoRA vs full fine-tuning on KAIST, Paderborn real damage, PHM 2010 | The sweeps (holiday queue) |

If Gate 2 finds no candidate better than late aggregation on the validation units, Plan B
becomes the model (plan, Gate 2), and weeks 8-9 pretrain its per-modality adapters instead.

Carried into Gate 2 from the audit (`PROJECT_LOG.md`, week 5):
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
