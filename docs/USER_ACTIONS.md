# Steps only you can take

Everything here needs your accounts, your credentials, a cluster session or a decision. The
code, checks and tests behind each step are ready, and each step says what success looks
like. Commands for the laptop assume the development environment is loaded first:

```powershell
. .\scripts\env\dev.ps1      # PDMFM_DATA_ROOT=D:\Project14\data, activates D:\Project14\.venv
```

| # | Step | Unblocks | Effort |
|---|---|---|---|
| 1 | Commit and push the current work | Ruche clones the repository from GitHub; CI runs | 10 min |
| 2 | Run the Ruche runbook (`docs/RUCHE_RUNBOOK.md`) | MOMENT, AST and PatchTST caches; Paderborn; the full Gate 1 | about 2 h of your time over a few days, the rest is queue and compute |
| 3 | Obtain PHM 2010 (manual download) | Milling wear-transfer target (Goal 2); milling physics score | 15-30 min |
| 4 | Three decisions (end of this page) | Scope of the next weeks | — |
| — | Done: GitHub repository, CI, pre-commit hook | — | — |

---

## 1. Commit and push the last fixes

Your commit "Stage 1 Implemtation" (`1a53d64`) holds most of this phase. The working tree has
the final round on top:

- **One test fixed for pandas 3.1**, which CI installs: text columns now have a string dtype,
  and `test_windows.py` asserted `object`. **CI on `1a53d64` has most likely failed that test**
  on all four test jobs.
- **Type checks clean:** pyright reports 0 errors in a CI-like environment and in the full one.
- **Cache listing:** photo caches now show as complete.
- **Updated logs and docs:** `PROJECT_LOG.md` and `docs/CI.md`.

`git status` lists them. Commit and `git push`; the four test jobs should then be green.

Nothing has been committed for you.

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
6. **Step 5 (§16):** download the caches and results, then tell me. I write up Gate 1 from
   `runs/gate1/gate1-v1/`.

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

## 4. Decisions

- **A. Milling transfer target.** PHM 2010 if step 3 succeeds; otherwise NASA milling, with
  the report stating its limits; or drop milling transfer and keep Nonastreda in-domain only.
- **B. Paderborn route.** Ruche (runbook §12) or a VPN.
- **C. CWRU's role.** The EDA shows a 10-class handcrafted floor of 1.0 under the cross-load
  split, plus a recorder artifact that separates normal from fault on its own
  (`docs/EDA_REPORT.md`). My recommendation is to use CWRU for pretraining and Gate 1 only,
  never as evidence. Gate 1 keeps it as one of its four tasks, as the plan fixes.

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
