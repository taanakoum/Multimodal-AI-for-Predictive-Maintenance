# Steps only you can take

Everything here needs your accounts, your credentials, or a decision. The scripts, checks and
tests behind each step are ready, and each step says what success looks like. Every command
below assumes the development environment is loaded first:

```powershell
. .\scripts\env\dev.ps1      # PDMFM_DATA_ROOT=D:\Project14\data, activates D:\Project14\.venv
```

| # | Step | Unblocks | Effort |
|---|---|---|---|
| 1 | Obtain PHM 2010 (manual download) | Milling wear-transfer target (Goal 2); milling physics score | 15-30 min |
| 2 | Fetch Paderborn on Ruche (or through a VPN) | Paderborn adapter, real-damage transfer target, vibration + current + temperature fusion | about 1 h, mostly waiting |
| 3 | Run the Ruche network check | Decides how GPU jobs get weights and data | 5 min |
| 4 | Push your commits (the GitHub repository already exists) | Automatic tests on every push | 5 min |
| 5 | ~~Install the pre-commit hooks~~ done | The same checks locally before each commit | — |
| 6 | Commit the current work | — | your call |
| 7 | Three decisions (end of this page) | Scope of weeks 2-4 | — |

---

## 1. PHM 2010 milling data

**Why.** The plan's second milling target (wear regression, leave-one-cutter-out) and the
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
python scripts\eda\run_eda.py phm2010    # inventory; I extend the EDA once the layout is confirmed
```

**Success.**
- `verify` reports `0 missing, 0 mismatched`.
- Both acceptance tests pass. They check:
  - 315 cuts for each of c1, c4 and c6, each with a wear value in µm that grows from cut to cut;
  - 7 columns per cut file;
  - a dominant force line at a multiple of 173.3 Hz, which confirms the commonly quoted 10,400 rpm
    (tooth passing 520 Hz for 3 flutes).

  Then set `VERIFIED_ON_REAL_DATA = True` in `pdmfm/datasets/phm2010.py` and tell me.

**If a test fails.**
- **Cut count or wear check fails:** the mirror packages the data differently; send me the
  file listing (`Get-ChildItem -Recurse D:\Project14\data\extracted\phm2010 | Select -First 30`).
- **Spindle check fails:** its message names the measured line. `SPINDLE_RPM` is then wrong,
  and I correct it before the physics score uses it.

**If you cannot obtain it:** NASA milling is already downloaded and verified, but it is a weak
substitute: 250 Hz, RMS-processed signals, 16 cases. See decision A at the end.

---

## 2. Paderborn bearing data (via Ruche, or a VPN)

**Why.** Paderborn has 32 bearings with vibration, two motor currents and temperature, and
damage that is either artificial or real (fatigue). It supplies the plan's "artificial to real
damage" transfer test and the in-domain multimodal sensor benchmark. Its host refused
connections from the development machine. The reader is unverified.

**Disk and licence.** 32 RAR archives, several GB in total: check free space on `$WORKDIR`
and `D:` first. CC BY-NC 4.0: research use is fine; do not redistribute the files.

### Option A: Ruche (recommended)

On the Ruche login node, from your copy of this repository (clone it after step 4, or copy the
folder with `scp -r`):

```bash
bash scripts/ruche/setup_env.sh        # once: Python 3.11 venv in $WORKDIR/pdmfm-venv (uv), torch CUDA
source scripts/env/ruche.sh            # every session: data root, HF cache, venv
bash scripts/ruche/fetch_paderborn.sh  # downloads (resumable), then verifies the SHA-256 of every archive
echo $WORKDIR                          # note the path for the copy below
```

Back on Windows (PowerShell; OpenSSH `scp` is built in), replacing `<login>` and `<WORKDIR>`:

```powershell
scp -r <login>@ruche.mesocentre.universite-paris-saclay.fr:<WORKDIR>/pdmfm_data/raw/paderborn D:\Project14\data\raw\
```

### Option B: VPN

Connect through a university VPN, then on the development machine:

```powershell
python -m pdmfm.fetch get paderborn --workers 4
```

### Then, in both cases

```powershell
python -m pdmfm.fetch verify paderborn     # re-hashes against the manifest written at download: catches copy damage
python -m pdmfm.fetch extract paderborn    # uses WinRAR's UnRAR.exe (already installed)
python -m pytest -m data -k paderborn -rs  # acceptance tests
python scripts\eda\run_eda.py paderborn
```

**Success.**
- `verify` reports `0 mismatched`.
- Both acceptance tests pass. They check:
  - all 32 bearings, each with 4 settings × 20 measurements;
  - vibration and currents at 64 kHz, about 4 s long;
  - the speed channel matching each setting's rpm.

  Then set `VERIFIED_ON_REAL_DATA = True` in `pdmfm/datasets/paderborn.py`. Tell me the
  vibration channel's unit from the EDA inventory; the reader still leaves it blank.

---

## 3. Ruche network check

**Why.** If compute nodes have no internet, every weight and dataset must be fetched on the
login node first. `scripts/env/ruche.sh` already sets `HF_HUB_OFFLINE=1` inside jobs, which is
the safe default either way.

```bash
cd <repository on Ruche>
sbatch scripts/ruche/net_check.sbatch     # compute node -> ./net_check-<jobid>.out
bash scripts/ruche/net_check.sbatch       # login node, printed directly, for comparison
```

If `sbatch` rejects the partition, run `sinfo -s` and change the `#SBATCH --partition` line.

**Reading the result.** One `OK`/`FAIL` line per host (HTTP code; `000` = no connection), then:

| Last line on the compute node | Meaning | What to do |
|---|---|---|
| `VERDICT: all 9 hosts reachable` | Jobs could download | Nothing; keep offline mode for reproducibility |
| `VERDICT: no internet on …` | The expected case | Run `bash scripts/ruche/prefetch_weights.sh` on the login node once; all data via step 2 |
| `VERDICT: k of 9 reachable; failing: …` | Partial (proxy, firewall) | Send me the file |

If `paderborn` also fails on the **login** node, use option B of step 2.

---

## 4. GitHub repository and CI

The repository already exists: `origin` is
`github.com/taanakoum/Multimodal-AI-for-Predictive-Maintenance`, branch `master`, and your
"Project-14 Skeleton" commit is pushed. No new repository is needed. GitHub Actions runs only on
GitHub, so CI starts when you push your next commits.

1. Commit (step 6) and `git push`.
2. Open the **Actions** tab of the repository. The `CI` workflow starts on every push:
   - `Lint and format`;
   - then `Tests (ubuntu / windows, Python 3.11 / 3.12)`, four jobs;
   - and `Type check (pyright)`, which is advisory.

   `docs/CI.md` explains each job and what a red job means.
3. Optional: **Settings → Branches → Add rule** for `master`; require the `Lint and format` and the
   four `Tests` checks.
4. Optional, and not needed yet: the secret `OMNIFUSER_READ_TOKEN`. It becomes useful in
   week 6, when the C2F-N fusion candidate imports `omnifuser.models`.
   - Create a fine-grained token with read-only *Contents* access to `taanakoum/OmniFuser`.
   - Add it under **Settings → Secrets and variables → Actions**.
   - Without it, CI skips the OmniFuser install. No current test needs the install: the
     Nonastreda reader test uses a stand-in module.

**Costs.** Private repositories get 2,000 free Actions minutes a month, and Windows minutes
count double. One push runs 6 jobs. If minutes run short, limit the Windows legs to pull
requests (`docs/CI.md` explains how). Dependabot opens grouped update PRs weekly. The weekly
`Data sources` workflow checks that every download host still serves the files we used.

---

## 5. pre-commit hooks

**Done (2026-10-26):** the hook is installed in `.git\hooks\pre-commit`. It records the
interpreter `D:\Project14\.venv\Scripts\python.exe`, so it runs on every commit from any
terminal or from VS Code's Source Control panel.

`pre-commit` is installed only inside the project environment. A bare `pre-commit` therefore
works only after `. .\scripts\env\dev.ps1`; otherwise call it by its full path:

```powershell
D:\Project14\.venv\Scripts\pre-commit.exe run --all-files    # check every tracked file now
```

The hooks run ruff (lint and format, the same version as CI), `actionlint`, YAML/TOML checks,
whitespace fixes, and a 1 MB limit that keeps data out of git.

---

## 6. Commit

Nothing has been committed for you. `git status` lists the work since your "Project-14 Skeleton"
commit. One reasonable split:

1. `pdmfm/` and `tests/`: schema, preprocessing, readers, resources, fetch `verify`, and their tests.
2. `scripts/eda/`, `docs/EDA_REPORT.md`, `docs/eda/`: the EDA and its report (the figures total
   about 1 MB).
3. `.github/`, `.pre-commit-config.yaml`, `scripts/ci/`, `docs/CI.md`: CI.
4. `scripts/ruche/`, `scripts/env/`, `docs/USER_ACTIONS.md`, `PROJECT_LOG.md`, `README.md`.

Run outputs, coverage files and `net_check-*.out` are git-ignored.

---

## 7. Decisions

- **A. Milling transfer target.** PHM 2010 if step 1 succeeds; otherwise NASA milling, with
  the report stating its limits (250 Hz, RMS-processed, 16 cases); or drop milling transfer
  and keep Nonastreda in-domain only.
- **B. Paderborn route.** Ruche (option A) or a VPN (option B).
- **C. CWRU's role.** The EDA shows a 10-class handcrafted floor of 1.0 under the cross-load
  split, plus a recorder artifact that separates normal from fault on its own
  (`docs/EDA_REPORT.md`, CWRU). My recommendation is to use CWRU for pretraining and as a
  sanity check only, never as evidence.
