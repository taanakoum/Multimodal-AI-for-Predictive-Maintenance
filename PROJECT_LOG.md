# Project 14 — project log

This is the running record of tasks, progress, findings, implementations and decisions. Each week has its own section, newest last. The plan it follows is the [Project 14 implementation plan](https://claude.ai/artifact/1M2DPuffJP1mRt2oiUujx5); a reference copy is in [`docs/PLAN.md`](docs/PLAN.md).

## Status at a glance (updated 2026-10-29)

| Area | State |
| --- | --- |
| Phase | Stages 1-4 of the plan (weeks 2-5) implemented and tested. Stage 4 (Plan B: per-modality evidential experts, Dempster's rule and MoNIG) run locally on Nonastreda and XJTU-SY (week 5). The GPU part and Paderborn are ready for Ruche (`docs/RUCHE_RUNBOOK.md`, §13-§16) |
| Tests | See "Week 5" below for the counts; lint, format, pyright and pre-commit clean |
| CI | Runs on GitHub on pushes to `master` and on pull requests (`docs/CI.md`) |
| Datasets on `D:\Project14\data` | CWRU, KAIST, XJTU-SY, Nonastreda, Najafi thermal, NASA milling, MBHM metadata; window stores built for CWRU, KAIST, XJTU-SY, Nonastreda. **Blocked:** Paderborn (Ruche runbook §12), PHM 2010 (manual download) |
| EDA | `docs/EDA_REPORT.md`: 13 findings, three of them preprocessing rules |
| Needs you | `docs/USER_ACTIONS.md`: commit and push week 5; run the Ruche runbook (now with Plan B); PHM 2010; 4 decisions |

## Week 1 — setup, downloads and verification

### Objectives (from the plan)

- [x] New repository with licence (MIT), `.gitignore`, README, CI workflow
- [x] Python 3.11 environment outside OneDrive, with core, encoder, forecasting and XAI packages
- [x] Dataset registry and a resumable, hash-checked downloader
- [ ] All datasets local: 6 of 8 done (Nonastreda, CWRU, KAIST, XJTU-SY, Najafi, MBHM metadata); Paderborn and PHM 2010 blocked
- [ ] Week-1 verification checklist: 10 of 12 settled (2 need your input)
- [x] Encoder smoke test and CPU throughput benchmark (weights in `D:\Project14\hf_cache`, 4.1 GB)

### What was implemented

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Package `pdmfm`; Python ≥ 3.11, < 3.13; extras `encoders`, `forecast`, `xai`, `anomaly`, `dev` |
| `pdmfm/paths.py` | Data root from `PDMFM_DATA_ROOT` (default `<repo>/data`); `raw/`, `extracted/`, `interim/` folders |
| `pdmfm/sources.py` | Registry of the 8 datasets: download route, licence, citation, role. Also the CWRU file table, Paderborn bearing lists and settings, and XJTU-SY Drive file IDs |
| `pdmfm/fetch.py` | CLI `list`, `plan`, `get`, `extract`. Mendeley API folder walk, Hugging Face listing, Google Drive (gdown), direct HTTP with resume. Checks sizes and SHA-256 and writes `MANIFEST.json` per dataset, with each file's origin. `--workers N` downloads in parallel; `--seed-from DIR` reuses local copies that match the publisher's SHA-256; `--exclude TEXT` skips paths. Extracts zip, and RAR through Windows `tar` or 7-Zip |
| `tests/` | 19 offline tests: registry consistency, CWRU 10 classes × 4 loads, Paderborn split disjointness, downloads (hash, resume, HTML rejection, manifest, parallel workers, local seeds, exclusion) against a local HTTP server, Mendeley path assembly, extraction |
| `scripts/env/dev.ps1`, `dev.sh`, `ruche.sh` | Environment variables for the dev machine and Ruche |
| `scripts/week1/encoder_smoke.py` | Loads every planned encoder on CPU; records shapes, parameters, timings to `D:\Project14\logs\encoder_smoke.json` |
| `scripts/week1/inspect_datasets.py` | Format, rate, duration and image-size checks per dataset; writes `D:\Project14\logs\inspect_<name>.json` |
| `scripts/week1/encoder_throughput.py` | Batched CPU throughput per encoder, and the hours needed to cache the corpus |
| `scripts/ruche/net_check.sbatch` | 5-minute CPU job: can compute nodes reach Hugging Face, Mendeley, Paderborn, CWRU, PyPI? |
| `scripts/ruche/prefetch_weights.sh`, `fetch_paderborn.sh` | Run on the Ruche login node: cache all weights; fetch Paderborn |
| `.github/workflows/tests.yml` | Lint and tests on Ubuntu with CPU torch; runs once a GitHub remote exists |
| `docs/PLAN.md` | Copy of the plan as exported on 2026-10-25 |

### Environment

| Item | Value |
| --- | --- |
| Machine | Windows 11; Intel i7-11370H (4 cores, 8 threads); 16 GB RAM; one NVMe SSD (C: 366 GB with 87 GB free; D: 557 GB with 545 GB free); CPU only |
| Folders | Data `D:\Project14\data`, venv `D:\Project14\.venv`, weights `D:\Project14\hf_cache` and `D:\Project14\weights`, logs `D:\Project14\logs` |
| Python | 3.11.14 via uv |
| Packages | torch 2.14.1+cpu, transformers 5.20.2, peft 0.21.2, sentence-transformers 6.2.0, mantis-tsfm 1.1.0, momentfm 0.1.5 (GitHub), timesfm 3.0.2, captum 0.9.0, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, omnifuser (editable, `../OmniFuser`) |
| Git | Initialized; nothing committed yet (commits only on request) |

### Dataset status

| Dataset | Role | Size | Route | State |
| --- | --- | --- | --- | --- |
| CWRU | source | 134 MB, 40 files | Direct from engineering.case.edu | Done; verified (see findings) |
| Najafi thermal | optional | 11 MB | Mendeley API | Done; extracted with Windows `tar` |
| MBHM metadata | optional | 6.6 MB (data.hdf5 13 GB not fetched) | Hugging Face | Done; checked |
| Nonastreda (full) | source | 1.2 GB used (force, labels, tool, chip and workpiece photos) | Your `Data\Nonastreda_Data` reused for force, labels and the 512 tool photos, each matching Mendeley's SHA-256; the 512 chip and 512 workpiece photos downloaded. Precomputed spectrograms and scalograms skipped (whole-pass, leaky) | Done; 0 hash mismatches |
| KAIST | target | 4.25 GB in 3 zips (8.2 GB unpacked) | Mendeley API (SHA-256 per file) | Done; 45 conditions checked (see checklist item 7) |
| XJTU-SY | source | 4.4 GB in 6 RAR5 volumes; 12 GB of CSV once extracted | Google Drive via gdown; extracted with WinRAR's `UnRAR.exe` (Windows `tar` reads only the first volume) | Done; 15 bearings, 9,216 snapshots, each lifetime equal to the published one |
| Paderborn | source + target | 32 RAR | groups.uni-paderborn.de | **Blocked:** connection refused on ports 80 and 443 from this machine |
| PHM 2010 | target | ~800 MB × 6 | Original CDN | **Blocked:** cdn.teamholistic.net is now a domain-parking page (not followed); mirrors need an account |

### Week-1 checklist results

| # | Item | Result | Status |
| --- | --- | --- | --- |
| 1 | UniFault checkpoint | The "Tiny" link (tinyurl) leads to a Dropbox folder holding one Lightning checkpoint, `pretrain-epoch=1.ckpt` (9.8 MB, SHA-256 `d43adc79…5a0b58` for the zip). It loads with `torch.load(weights_only=True)`: 0.82 M parameters, d = 128, 16 patches of 64 samples, so 1,024-sample univariate input. Running it needs the model class from the MIT repo (week 3). Stored in `D:\Project14\weights\unifault` | Settled |
| 2 | `momentfm` pins | PyPI 0.1.4 pins transformers 4.33.3, numpy 1.25.2, torch ~2.0. GitHub main (0.1.5) needs transformers ≥ 4.54.1; it installs into the main environment and runs with transformers 5.20.2 (smoke test) | Settled |
| 3 | MBHM unit mapping | No unit (bearing) ID; conditions are dataset, bearing type, channel, rpm, load. 135,516 windows from 9 datasets. After dropping PU (Paderborn) and XJTU: **114,864 windows** from CWRU, DIRG, HIT, IMS, JUST, MFPT, NCEPU. Windows are 24,000 points, made by Discrete Cosine Normalization. MIT | Settled: use with dataset-level exclusion |
| 4 | XJTU-SY licence and geometry | No licence stated; the authors ask for a citation of Wang et al. 2020. LDK UER204: 8 balls, d = 7.92 mm, D = 34.55 mm, φ = 0°, so BPFO = 3.083 f_r and BPFI = 4.917 f_r. 25.6 kHz; 32,768 points (1.28 s) every minute; 2100 / 2250 / 2400 rpm | Settled |
| 5 | PHM 2010 details and access | Confirmed: 10,400 rpm, 1,555 mm/min feed, 0.125 mm radial / 0.2 mm axial depth, 6 mm ball-nose carbide cutter, 50 kHz, 7 channels (Fx, Fy, Fz in N; Vx, Vy, Vz in g; AE-RMS in V), wear per cut in 10⁻³ mm. Number of flutes not stated. Download blocked (see dataset status) | Open: needs your input |
| 6 | Paderborn facts | From the PHME 2016 documentation: healthy K001-K006 (Table 7), 12 artificial (Table 4), 14 real (Table 5); settings N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N), N09_M07_F10 (900 rpm), N15_M01_F10 (0.1 Nm), N15_M07_F04 (400 N); vibration and current at 64 kHz, auxiliary channels at lower, unstated rates (to read from the files); CC BY-NC 4.0. The paper's figure shows BPFO = 76.25 Hz at 25 Hz, which matches our 6203 geometry estimate | Settled (facts); download blocked |
| 7 | KAIST formats and DAQ synchronization | **Vibration** (`vibration.zip`, 45 MAT files): Siemens Test.Lab export, struct `Signal` with `y_values.values` of shape (n, 4) in g (4 housing accelerometers, Point1-4), exactly 25.6 kHz, absolute start time per file. **Current and temperature** (`current,temp.zip`, 45 TDMS files): group `Log` (the first group, `Test Information`, is empty), channels Mod1/ai0-ai1 thermocouples in °C and Mod2/ai0, ai2, ai3 currents in A, all at 1/39.05 µs ≈ 25.608 kHz. **Durations:** bearing faults 60 s, misalignment and unbalance 120 s, normal 300 s at 0 Nm and 120 s at 2 and 4 Nm (the article says 120 s and 60 s). **Synchronization:** in 40 of 45 conditions the two devices start 3.4 min apart with equal durations, consistent with one session and a clock offset between the two PCs. In 5 conditions (0Nm_BPFI_03, 0Nm_BPFI_10, BPFO_03 at 0, 2 and 4 Nm) they are 23-30 min apart, so not simultaneous. **Names:** the 2 Nm unbalance files are spelled `2Nm_Unbalalnce_*` in both zips; the reader normalizes them. Acoustic (5 files at 0 Nm) is unused, as planned | Settled |
| 8 | Nonastreda sampling rate | 1 kHz, stated in the dataset's own Mendeley description; licence CC BY 4.0 | Settled |
| 9 | Nonastreda photo sizes | Tool 1550×500 (T1 and most of T2: 3100×1000, same 3.1:1 aspect); chip 1024×1024; workpiece 512×512. DINOv2 inputs: tool 224×672 → class token + 2×4 pooled patches = 9 tokens; chip and workpiece 224×224 → class token + 2×2 = 5 tokens each, so at most 19 image tokens per sample (the plan said 27) | Settled |
| 10 | Ruche compute-node internet | Needs one job from you: `sbatch scripts/ruche/net_check.sbatch` | Open: needs your input |
| 11 | Code licences | MIT: UniFault, MOMENT, Chefer relevance, OGM-GE, perceiver-pytorch. Apache-2.0: Mantis, TimesFM, DINOv2, anomalib, PHM-Vibench. BSD-3-Clause: Captum. **No licence:** TMC (now Han-Zongbo/TMC), MoNIG, BearLLM. AnomalyGPT: NOASSERTION | Settled: rewrite TMC and MoNIG from their papers |
| 12 | Najafi thermal set | 369 images in 11 condition folders (A10, A30, A50, A&C10, A&C30, A&B50, A&C&B10, A&C&B30, Fan, Noload, Rotor-0) plus 40 annotated images with ground truths; all 320×240 RGB pseudo-colour; Dali-tech T8 camera (384×288 detector) | Settled |

### Findings that refine the plan

1. **CWRU 99.mat holds a copy of 98.mat's signal** (`X098_DE_time`, identical) next to its own `X099_DE_time`. The CWRU reader must select variables by the file's own number, or the 1 hp normal recording would appear twice. Files 98 and 99 have no RPM field (use the nominal 1772 and 1750 rpm). File 97 is only 5.1 s long; all fault files are about 10.1 s.
2. **MBHM is still useful without unit IDs.** Dropping its Paderborn and XJTU-SY subsets leaves 114,864 vibration windows from seven datasets, none of them our targets. That is several times our own vibration corpus, and it feeds Level-1 encoder pretraining (own PatchTST) only, never evaluation.
3. **Paderborn's original protocol** (PHME 2016, Table 8) trains on K002, KA01, KA05, KA07, KI01, KI05, KI07 and tests on K001, KA04, KA15, KA16, KA22, KA30, KI14, KI16, KI17, KI18, KI21. Our target split contains this test set, so we can also report on exactly these bearings for comparability.
4. **RAR extraction:** Windows' built-in `tar` extracts single-volume RAR (Najafi), but stops at the end of the first volume of a multi-volume RAR5 archive (XJTU-SY). The extractor therefore prefers 7-Zip, then WinRAR's `UnRAR.exe`, which is already installed on this machine, and falls back to `tar` last. XJTU-SY CSVs (`Horizontal_vibration_signals,Vertical_vibration_signals`, 32,768 rows) take 12 GB as text. Week 2 converts them to binary shards, after which the extracted CSVs can be deleted (the RAR parts stay as the source of truth).
5. **All seven encoders run with transformers 5.20.2 on CPU** (`D:\Project14\logs\encoder_smoke.json`):

    | Encoder | Parameters | Input → output | Licence |
    | --- | --- | --- | --- |
    | Mantis-8M | 8.1 M | (B, 1, 512) → (B, 256) | Apache-2.0 |
    | MOMENT-1-base | 109.6 M | (B, 1, 512) → (B, 768) | MIT |
    | DINOv2-S/14 | 22.1 M | (1, 3, 224, 672) → (1, 769, 384) tokens | Apache-2.0 |
    | Qwen3-Embedding-0.6B | 595.8 M | sentences → (n, 1024) | Apache-2.0 |
    | AST (AudioSet) | 86.2 M | 16 kHz audio padded to 1024 frames × 128 mel bins → (1, 768) | BSD-3-Clause |
    | TimesFM 3.0 | 330.7 M | context 512 → 32-step forecast + 9 quantiles | non-commercial weights |
    | TimesFM 2.5 | 231.3 M | context 512 → 32-step forecast + 10 quantile outputs | Apache-2.0 |

6. **CPU throughput decides what can be cached locally** (`encoder_throughput.json`, 8 threads, batch 64-256):

    | Encoder | CPU rate | Whole-corpus T arm (≈113k channel-windows × 24 segments) | F and E arms |
    | --- | --- | --- | --- |
    | Mantis-8M | 279 windows/s | 2.7 h | 0.2 h |
    | MOMENT-base | 19 windows/s | **40 h** | 3.3 h |
    | AST | ≈ 0.8 windows/s (single-window smoke test; AST pads every input to 10.24 s) | ≈ 40 h for one spectrogram per window | - |
    | DINOv2-S | 5.4 photos/s (224×672) | all 1,543 photos in 5 min | - |

    Consequence: Gate 1 runs locally on CPU with Mantis on all arms, and with MOMENT (T arm) and AST on a unit-stratified 10 % subsample of the probe tasks. If MOMENT or AST wins Gate 1, the full cache is built on a Ruche GPU. AST should also get a shorter `max_length` (about 100 frames for 1 s inputs), with its positional embeddings interpolated; that change belongs to week 3.
7. **Fine-grained fault descriptions sit close together in Qwen3 space:** cosine 0.966 between "outer-race defect 1.0 mm" and "inner-race defect 1.0 mm", 0.83 to "healthy". Zero-shot separation will rely on the learned sensor-side projection and on paraphrase ensembles, as planned. Raw prompt similarity alone will not separate these classes.
8. **Your local Nonastreda copy is the Mendeley release.** Force file and tool photos match the published SHA-256 exactly. `labels.csv` differs only by Windows line endings (513 CRLF; identical after normalization).
9. **Mendeley rejects requests without a browser-like User-Agent** (HTTP 403); the downloader sets one.
10. **KAIST modalities are paired by condition, not by sample.** The two acquisition PCs' clocks differ by a constant 3.4 min. The sample rates also differ slightly (1,536,000 samples at 25.6 kHz against 1,536,492 at about 25.608 kHz over the same 60 s), so samples do not line up one to one. And 5 of 45 conditions were not recorded at the same time. So a vibration window and a current or temperature window from the same condition are a valid pair of the same machine state, but not a time-synchronous one. This matches the plan's rule (align KAIST by recording only). The paper and tables must describe KAIST fusion as condition-level.
11. **KAIST test-set size at 4 Nm:** about 1,400 windows of 1.024 s (6 bearing-fault conditions × 58 + 9 other conditions × 117), all 15 classes present.

### Needs your input

1. **PHM 2010** (held-out milling target): the original download host is gone. Options:
    - (a) create or use a Hugging Face account, accept the terms of `AI4Manufacturing/PHM2010`, and give me a read token (or download c1, c4, c6 yourself into `D:\Project14\data\raw\phm2010`);
    - (b) use IEEE DataPort or Kaggle;
    - (c) replace PHM 2010 with the NASA milling dataset.
2. **Paderborn** (source and target): the host refuses connections from this machine. Options:
    - (a) run `scripts/ruche/fetch_paderborn.sh` on the Ruche login node and copy the files back;
    - (b) download through a VPN, after which `python -m pdmfm.fetch get paderborn` works as is.
3. **Ruche network test:** submit `sbatch scripts/ruche/net_check.sbatch` once and paste the output, to settle checklist item 10.

### Next steps

Rest of week 1, as soon as you reply:
- PHM 2010 by the route you choose;
- Paderborn via Ruche or a VPN, then the same checks as for KAIST (rates of the auxiliary channels read from the files);
- the Ruche network result.

Week 2 (Nov 2-8, plan):
1. `pdmfm/schema.py` (`ChannelWindow`, `ImageItem`, `Sample`, as-of join) with its tests.
2. Readers for Nonastreda (on top of `omnifuser.data.load_nonastreda(pool=1)`), CWRU (own-number variables only), KAIST (normalized names, `Log` group, condition-level pairing), XJTU-SY (CSV to binary shards) and the Najafi set.
3. `pdmfm/preprocess.py`: 12 kHz anti-aliased resampling, windows, T/F/E inputs, level and slow descriptors, healthy baselines.
4. Context and fault-description texts per dataset (`pdmfm/texts/*.yaml`).
5. Leakage tests: unit-disjoint splits; no item captured after its window starts.

### Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-25 | Data in `D:\Project14\data`, venv in `D:\Project14\.venv`, weights in `D:\Project14\hf_cache` | Outside OneDrive; 545 GB free on the internal NVMe |
| 2026-10-25 | Develop and validate locally on CPU; Ruche only for final training | Your instruction |
| 2026-10-25 | MIT licence; git initialized without commits | Your choice; commits only on request |
| 2026-10-25 | `momentfm` from GitHub into the main environment | PyPI release pins outdated libraries |
| 2026-10-25 | MBHM used without its PU and XJTU subsets, for encoder pretraining only | No unit IDs to exclude held-out bearings |
| 2026-10-25 | Rewrite the TMC and MoNIG fusion rules from the papers | Their repositories have no licence |
| 2026-10-25 | Nonastreda: reuse your `Data\Nonastreda_Data` (hash-verified), download only chip and workpiece photos, skip precomputed spectrograms and scalograms | Your request; spectrograms and scalograms cover whole passes and are not used by the plan |
| 2026-10-25 | KAIST fusion treated as condition-level pairing (not sample-synchronous) | Clock offset between the two devices and 5 non-simultaneous recordings |
| 2026-10-25 | Gate 1 on CPU: Mantis on all arms; MOMENT T arm and AST on a 10 % unit-stratified subsample | Full MOMENT T-arm or AST caching takes about 40 h on this CPU |

## Week 1, continued (2026-10-26): data layer, EDA, CI

### What can be implemented and tested now, on this CPU

| Work | Status | Why it is reliable locally |
| --- | --- | --- |
| Sample schema and as-of join (`pdmfm/schema.py`) | Done, tested | Pure logic |
| Readers for the 7 local datasets (`pdmfm/datasets/`) | Done; tested on synthetic files and on the real data | Data on disk |
| Shared preprocessing (`pdmfm/preprocess.py`) | Done, tested; 5 kHz common band and tachless speed added after the EDA | Numerical tests on synthetic signals |
| CPU/RAM guard (`pdmfm/resources.py`) | Done, used by the EDA | — |
| EDA and report | Done (`docs/EDA_REPORT.md`) | Full run about 5 min |
| Next: context and fault-description texts (`pdmfm/texts/*.yaml`) | Not started | No compute |
| Next: unit-level split protocols (`pdmfm/protocols.py`) | Not started | Pure logic |
| Next: feature cache, filled with Mantis (279 windows/s) and DINOv2 (5.4 images/s) | Not started | All local datasets in under an hour |
| Next: Gate-1 encoder probe | Not started | Mantis on every arm; MOMENT and AST on a 10 % unit-stratified subsample (full T-arm caching takes about 40 h on this CPU) |
| Next: Plan B (per-modality evidential heads) and a small fusion search on cached features | Not started | Small tensors |
| Blocked: Paderborn and PHM 2010 adapters | Infrastructure ready | Data (your steps 1-2) |
| Later on Ruche: full MOMENT/AST caching, PatchTST self-pretraining, fusion pretraining, adaptation sweeps | — | Too slow for CPU |

### What was implemented

| File | Purpose |
| --- | --- |
| `pdmfm/schema.py` | `ChannelWindow`, `ImageItem`, `Sample`; as-of join (newest item strictly before the window start) |
| `pdmfm/preprocess.py` | Exact-ratio resampling to 12 kHz, **5 kHz common band (`band_limit`, `model_input`)**, windows, T/F/E arms, **`dominant_frequency`**, level and slow descriptors, healthy baselines, defect frequencies, FPT, normalized RUL |
| `pdmfm/datasets/` | Readers: CWRU, KAIST, XJTU-SY, Nonastreda, Najafi, NASA milling; Paderborn and PHM 2010 written against their documentation (`VERIFIED_ON_REAL_DATA = False`) |
| `pdmfm/resources.py` | Checks idle CPUs, free RAM and busy processes before heavy jobs; caps threads (torch, BLAS); aware of SLURM |
| `pdmfm/fetch.py` | New `verify` command: re-hashes against `MANIFEST.json`, records hand-placed files (PHM 2010), checks required files |
| `scripts/eda/run_eda.py` | EDA of every dataset on disk; summaries and figures in `docs/eda/` |
| `.github/workflows/tests.yml`, `sources.yml`, `dependabot.yml`, `.pre-commit-config.yaml` | CI on every push; weekly data-source check; dependency updates; local hooks (`docs/CI.md`) |
| `scripts/ci/run_local_ci.sh`, `scripts/ci/check_sources.py` | CI reproduced in a fresh environment; host reachability table |
| `scripts/ruche/setup_env.sh`, `net_check.sbatch` (rewritten), `fetch_paderborn.sh` (verifies after download) | Your Ruche steps (`docs/USER_ACTIONS.md`) |
| Tests | `test_schema`, `test_preprocess`, `test_readers`, `test_resources`, `test_eda`, `test_real_data` (regression and acceptance), `test_network`, `test_fetch` (+ `verify`) |

### Findings (details and numbers in `docs/EDA_REPORT.md`)

1. **CWRU recorder artifact.** The 48 kHz normal files and the 12 kHz fault files differ above
   5.25 kHz; that band alone separates normal from fault (balanced accuracy 1.0). Fixed for all
   sources by a 5 kHz linear-phase band limit and by spectra that end at 5 kHz.
2. **KAIST sessions are confounded with class.** Normal, unbalance and misalignment were recorded
   on 25 June, inner race on 28 June, outer race on 30 June. Temperature follows the session and
   is constant within a recording, so it is excluded as an absolute input.
3. **KAIST current fundamental (48.5 Hz) sits below the shaft rate (50.17 Hz).** That is
   consistent with the electrical device's stated sample rate being about 3.4 % low (open
   question).
4. **KAIST cross-load floor.** Vibration 0.526 (15 classes), 0.734 (5 types); current and
   temperature at chance.
5. **XJTU-SY RUL labels.** The default 3σ FPT fires at 0.6 % of life on Bearing3_1. A baseline of
   10 % of life, 3σ and 10 consecutive exceedances gives no early alarms and an onset for 14 of 15
   bearings; it is recommended for offline labels.
6. **XJTU-SY envelope lines.** Outer-race failures show their line in every case; inner-race
   failures do not. The physics score there is restricted to the outer race.
7. **Nonastreda baselines.** Label persistence is 78.8 %, above the published ~75 %, so it goes in
   every table. The leave-one-tool-out floor from Fx/Fy levels is 0.566. Fz drifts by block.
8. **Nonastreda 66.665 Hz line.** Constant in all passes and phase-locked over 15 samples; most
   consistent with a 4,000 rpm spindle and a 4-flute cutter, with aliased tooth harmonics
   (unconfirmed).
9. **Najafi.** Near-duplicate images, trivially separable (1.0 under any split): never evidence.

### Infrastructure for your steps

- **PHM 2010.** Drop the zips in the raw folder; `python -m pdmfm.fetch verify phm2010`;
  `extract`; acceptance tests check 315 cuts, wear in µm, 7 columns and the 10,400 rpm spindle
  line.
- **Paderborn.** `scripts/ruche/setup_env.sh` and `fetch_paderborn.sh` on Ruche; `scp` back;
  `verify` catches copy damage; acceptance tests check 32 bearings × 80 files, 64 kHz and the
  setting speeds. The acceptance tests themselves were dry-run against a synthetic layout built to
  the documentation (4 passed).
- **Ruche network check.** `net_check.sbatch` writes next to the submit directory (it no longer
  needs `logs/`), covers 9 hosts and ends with a verdict line. It was tested locally: 8 of 9
  hosts reachable, Paderborn refused.

### Additional decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-26 | Every sensor channel goes through `pp.model_input` (resample to 12 kHz, band-limit to 5 kHz); no log-domain feature above 5 kHz | Recorder roll-offs identify the source and, on CWRU, the class (EDA F1) |
| 2026-10-26 | E arm uses the shaft speed measured from vibration per recording | Nominal speeds are approximations; CWRU headers agree within 0.12 %, so the method is validated |
| 2026-10-26 | KAIST: no absolute temperature input; `current_1` only; splits by condition and load | Session confound, recording-level constants, class-dependent empty channels |
| 2026-10-26 | XJTU-SY RUL labels from the 10 %-of-life FPT rule (kurtosis fallback for abrupt failures) | Default rule raises false early alarms on long runs |
| 2026-10-26 | Nonastreda tables always include the persistence baseline | Persistence (78.8 %) beats the published accuracy |
| 2026-10-26 | Ruff pinned to 0.17.2 in pyproject, pre-commit and CI | Formatting differs between versions |
| 2026-10-26 | Nothing committed | Your instruction: you commit yourself |

## Weeks 2-4 (2026-10-28/29): stages 1-3 of the pipeline

### What was implemented

| File | Purpose |
| --- | --- |
| `pdmfm/texts/*.yaml`, `pdmfm/texts/__init__.py` | Context templates (machine, sensors, speed, load; never the state) and 3 paraphrases per class (never speed or load) for 7 datasets; class keys `label` or `label/severity` |
| `pdmfm/protocols.py` | The plan's splits as row masks: 5 outer folds (Nonastreda, XJTU-SY, Paderborn's 15 source bearings in stratified triples), Nonastreda leave-one-tool-out, CWRU Gate-1 split, KAIST/Paderborn/PHM 2010 targets, few-shot draws, unit-stratified subsamples, pretraining masks per fold |
| `pdmfm/windows.py` | Stage 1: a window store per dataset (band-limited model input in float32, levels, slow summaries, measured shaft speed, labels, context, previous-pass photos, XJTU-SY RUL), written atomically, refused when built by other code |
| `pdmfm/arms.py` | Encoder inputs T (512-sample segments), F (log spectrum), E (envelope orders), S (waveform) |
| `pdmfm/encoders.py` | One interface for the encoder bank: handcrafted reference, Mantis, MOMENT, own PatchTST, AST (positions resized to 100 frames), DINOv2 photo tokens, Qwen3 text |
| `pdmfm/features.py` | Stage 2: resumable feature caches keyed by store, checkpoint and arm; photo and text caches |
| `pdmfm/patchtst.py` | Own PatchTST masked-patch pretraining per outer fold; resumable |
| `pdmfm/probe.py` | Stage 3: Gate 1 on the plan's four tasks, inner-validation units only, shared rows, per-fold models, parallel fits |
| `pdmfm/utils.py` | Stable hashes and atomic writes |
| `slurm/` | Ruche: environment, GPU/CPU job wrappers, step-0 checks, chains, status, network check, login-node downloads (replaces `scripts/ruche/` and `scripts/env/ruche.sh`) |
| `docs/RUCHE_RUNBOOK.md` | Exact commands for the Ruche part, after the OmniFuser runbook |
| Tests | `test_texts`, `test_protocols`, `test_windows`, `test_encoders`, `test_features`, `test_probe`, `test_patchtst`, `test_slurm`; new real-data and regression tests; `tests/synthetic.py` (readers in each dataset's layout) |

### Problems found and fixed in this phase

| Problem | How it was found | Fix |
| --- | --- | --- |
| The first window of every Nonastreda pass never received the previous pass's photo: passes were laid end to end and photos stamped exactly at the next pass's start | Window-store test | A photo pause between passes on each tool's clock (`PASS_GAP_S`); on the real store all 4,134 later-pass windows now carry all three photos, the 107 first-pass windows none |
| The handcrafted encoder's band means diluted narrow spectral lines (a tone moved its band mean by 0.07) | Probe test on synthetic tones | Band maxima added (36 dimensions) |
| `unrar` destination ended with a backslash, wrong on Linux (Ruche) | Writing the runbook | `os.sep`; tested |
| DINOv2 wrapper needed a processor config absent from an offline cache; Qwen3 loaded in bfloat16 on CPU (slow, norms off by 2e-3) | Real-checkpoint tests offline | ImageNet constants; float32 |
| Manifest columns `meta_*` read back as text (minute 10 sorted before minute 2) | Window-store tests | Numeric parsing for all non-text columns |
| Gate 1 skipped a whole task when a candidate lacked an arm (E on Nonastreda), and could name a reference as winner | Planning the runbook commands; reading the code | Arms a dataset lacks are dropped; a candidate with none takes the worst rank; winner chosen among candidates only |
| Review (code-review skill, high): stale stores never refused; `--limit` smoke builds replaced the full store; a missing reference skipped a task; several caches of one encoder blocked the probe; an XJTU-SY snapshot shorter than a window shifted RUL labels; an empty dataset crashed the PatchTST corpus sampler; dataset names hard-coded in the builder; duplicated helpers; two extra corpus copies | Review of the uncommitted diff | All fixed and tested. The tenth point (the photo-pause change without a version bump) needed no bump: the store version was introduced in the same change set and every store was built after it; the rule is now documented next to `WINDOWS_VERSION` |

### Local runs (CPU)

| Run | Result |
| --- | --- |
| Window stores (`python -m pdmfm.windows build cwru nonastreda kaist xjtu_sy`) | 356, 4,241, 4,378 and 9,216 windows in 6 min; 2.4 GB in total; counts equal the EDA's. XJTU-SY labels follow the 10 %-of-life FPT rule (Bearing3_1 at 0.924 of life; Bearing1_4 by kurtosis) |
| Feature caches (handcrafted and Mantis on T, F, E; CWRU, XJTU-SY, Nonastreda) | Mantis: about 206 inputs/s on 4 threads, 55 min for every arm (the T arm dominates: 48 inputs per XJTU-SY window) |
| Photos and texts | DINOv2: 512 tool photos (9 tokens) in 118 s, 512 chip and 512 workpiece photos (5 tokens) in 52 s and 39 s, 369 thermal images in 27 s. Qwen3: contexts per load or condition, 3-15 class prototypes per dataset |
| MOMENT and AST on the 10 % unit-stratified subsample | MOMENT about 19 inputs/s (1 h 10 min); AST 5.8 windows/s with 100-frame inputs (3 min) |
| Gate 1, preliminary, all local data (`runs/gate1/gate1-cpu-full/`) | See below. 1.5 min for 7 entries x 3 tasks |
| Gate 1, preliminary, 10 % subsample (`runs/gate1/gate1-cpu-subsample/`) | All 12 entries on the common subsample; too few validation windows to rank (CWRU: about 10 windows for 10 classes), recorded for completeness only |

### Gate 1, preliminary (CPU; Paderborn absent, so not the gate decision)

Balanced accuracy (CWRU 3 hp; Nonastreda inner-validation tools, mean of 5 folds) and RMSE of
normalized RUL (XJTU-SY inner-validation bearings, mean of 5 folds); all local windows:

| Candidate | CWRU | Nonastreda | XJTU-SY (RMSE) | Mean rank |
| --- | --- | --- | --- | --- |
| mantis:F | 1.000 | 0.662 | 0.228 | 2.00 |
| mantis:E | 0.967 | (no E arm) | 0.195 | 2.33 |
| mantis:T+E | 0.933 | 0.595 | 0.208 | 2.50 |
| mantis:T | 0.911 | 0.595 | 0.226 | 3.17 |
| stats:F (handcrafted reference) | 1.000 | 0.667 | 0.184 | – |
| stats:E (handcrafted reference) | 0.989 | (no E arm) | 0.183 | – |
| stats:T (handcrafted reference) | 0.756 | 0.539 | 0.224 | – |

Reading:
- **Mantis does not beat the handcrafted reference here.** It ties on CWRU, is level on
  Nonastreda and is worse on XJTU-SY RUL.
- The spectrum arms (F, E) beat the raw-time arm (T) on every task.
- CWRU is saturated (balanced accuracy 1.0 even for the handcrafted features), as the EDA predicted.
- The decision waits for the full MOMENT, AST and per-fold PatchTST caches and for Paderborn
  (Ruche runbook). If no frozen encoder clearly beats the references there either, that is a
  finding for the report: it supports the plan's Gate-1 fallback (own PatchTST) and argues for
  keeping the handcrafted level and band tokens next to the encoder tokens.

### Verification at the end of the phase

| Check | Result |
| --- | --- |
| Full suite with the real data and the cached checkpoints | 229 passed, 24 skipped (5 network, 4 Paderborn/PHM 2010 acceptance, 15 Linux-only SLURM tests) |
| CI reproduced in fresh Python 3.11 and 3.12 environments (`scripts/ci/run_local_ci.sh`) | 204 passed, 45 skipped (data, weights, SLURM), coverage 80 % (the encoder wrappers need the optional libraries, tested locally); pyright 0 errors |
| Fresh-environment finding | pandas 3.1 (what CI installs) gives text columns a string dtype; one test asserted `object`. Fixed after commit `1a53d64`, so CI on that commit most likely failed that test |
| pre-commit, all files | All 9 hooks pass |

## Week 5 (2026-10-29): stage 4, the Plan B baseline

Plan B is the plan's baseline, its fusion candidate 1 (late-evidential) at Gate 2, and the
fallback model. It is implemented, tested and run locally on Nonastreda and XJTU-SY. Paderborn
and the final Gate-1 encoder run on Ruche (runbook §16).

### What was implemented

| File | Purpose |
| --- | --- |
| `pdmfm/losses.py` | Rewritten from the papers (their repositories carry no licence). TMC: opinions from evidence, Dempster's rule with conflict, the expected cross-entropy plus annealed KL loss. Deep evidential regression: NIG parameters, NLL and regularizer, Student-t predictive. MoNIG summation, skipping absent modalities |
| `pdmfm/metrics.py` | Accuracy, balanced accuracy, macro-F1; ECE (15 bins), Brier, NLL; temperature scaling fitted on validation NLL; RMSE, MAE, 90 % interval coverage and width; PHM 2012 score |
| `pdmfm/model/inputs.py` | Token sets per modality from the caches, row-aligned with the window store. One expert per channel family (force, vibration, current), with level tokens relative to the unit's healthy baseline. A slow-channel expert (Paderborn temperature: mean and slope relative to the same baseline). A photo expert (previous pass's tool, chip and workpiece tokens with their age; masked when absent, never zeros). Text prototypes checked against the YAML order |
| `pdmfm/model/layers.py`, `model/late.py` | Modality encoder: adapter, role, channel, level and age embeddings, attention pooling with one learned query. Text-anchored evidential head (cosine to the frozen Qwen prototypes, softplus evidence) and NIG head. `LateEvidential` combines them by Dempster's rule and MoNIG |
| `pdmfm/train.py` | Training on cached features: TMC on each present modality and on the combination (KL weight min(1, epoch/10)); NIG loss on each modality and on the fusion, on standardized targets. AdamW 1e-3, 50 epochs. The epoch kept is the best on the inner-validation units by mean(bag macro-F1, 1 − RMSE/std). Bags (passes, recordings) average their windows' evidence and NIG parameters |
| `pdmfm/planb.py` | Stage 4 runner: 5 outer folds × 3 seeds; temperature fitted on the inner-validation units; results per fold and seed (atomic, resumable, refuses changed settings), then `summary.csv` and `summary.md`. Reports the combined model before and after calibration, each modality alone, the model without each modality, trust per modality, and the majority, persistence and training-mean baselines. Per-fold inputs where they depend on the fold (Paderborn's baseline, `patchtst-fold{fold}`); `--without` leaves modalities out |
| `slurm/status.sh` | Counts Plan B results per run (`15 of 15` when complete) |
| `docs/RUCHE_RUNBOOK.md` | New §16 (Plan B with the Gate-1 winner, the handcrafted reference, Paderborn, the temperature check); download and operating sections renumbered |
| Tests | `test_evidential` (23 cases), `test_metrics` (12), `test_inputs` (22), `test_planb` (23), one new `test_slurm` test; `tests/synthetic.py` gains a Paderborn reader and the cache builders |

### Design decisions

| Decision | Reason |
| --- | --- |
| One expert per channel family, one for slow channels, one for photos | The plan's "per-modality evidential heads"; channels of one family share a sensor type and rate |
| Context text is not an expert | It never names the state (by construction), so its opinion would be vacuous; it stays for the leakage check and the fusion stages |
| An absent modality has zero evidence, and its NIG is skipped | A vacuous opinion is the identity of Dempster's rule (tested), so missing data needs no imputation |
| Paderborn's level baseline is the healthy *training* bearings at the same setting, per fold | The plan's rule; a test bearing never enters its own reference |
| Persistence = the previous bag's true label or wear of the same unit | The plan's baseline; it assumes the last inspection is known, so it is a strong reference rather than a deployable model (amendment in `docs/PLAN.md`) |
| Window-level temperature, applied to bag probabilities as well | One parameter fitted on the inner-validation windows, as the plan says |

### Problems found and fixed in this phase

| Problem | How it was found | Fix |
| --- | --- | --- |
| Paderborn had no level-baseline rule, and slow channels were no modality: Plan B would have stopped on Ruche | Writing the runbook step | Fold-aware baseline (`train_units`), slow-channel expert with a small MLP embedding (a LayerNorm over two numbers would erase them); tested on a synthetic Paderborn store |
| A per-fold winner (`patchtst-fold{fold}`) could not run in Plan B | Same | Inputs assembled per fold when the encoder or the baseline depends on it |
| The channel embedding was sized from token channels only; a level channel without tokens would have indexed past it | Review | Sized from token and level channels; tested |
| `bag_mean` found each bag's first row in O(n·bags) | Review | `np.minimum.at`, O(n) |
| A learning test passed or failed with the random initialization (12 test bags) | Adding the slow role changed the initialization | Three seeds at lr 3e-3; checks the mean and every seed against the baselines |
| 12 pyright findings (Optional subscripts, numpy and pandas stubs, a buffer type) | `pyright pdmfm` | Fixed; 0 errors |
| Paderborn temperature may follow the recording session, as on KAIST (EDA finding 2) | Review against the EDA | `--without slow`, and a check in runbook §16.3 |

### Local runs (CPU, 5 outer folds × 3 seeds, test units, mean ± std)

All runs used AdamW 1e-3, 50 epochs and d = 256; results are in `runs/planb/<run-id>/`.

**Nonastreda** (leave-two-tools-out; a bag is a pass; 3 wear classes and flank wear in µm):

| Run | Bag macro-F1 | Bag balanced accuracy | Window ECE, raw → calibrated | Wear RMSE (µm) | 90 % interval coverage |
| --- | --- | --- | --- | --- | --- |
| Mantis F + photos (`planb-nona-mantis-F`) | **0.781 ± 0.074** | 0.782 ± 0.068 | 0.244 → 0.108 | 23.3 ± 11.0 | 0.69 |
| Handcrafted F + photos (`planb-nona-stats-F`) | 0.776 ± 0.089 | 0.782 ± 0.077 | 0.239 → 0.112 | 23.5 ± 10.9 | 0.69 |
| Mantis F, no photos (`planb-nona-mantis-F-nophoto`) | 0.532 ± 0.134 | 0.599 ± 0.113 | 0.156 → 0.144 | 33.0 ± 9.5 | 0.70 |
| Persistence (previous pass's label and wear) | 0.756 ± 0.106 | 0.755 ± 0.101 | – | **20.6 ± 9.4** | – |
| Majority class / training mean | 0.158 ± 0.041 | 0.333 | – | 44.6 ± 11.0 | – |

Inside the Mantis + photos model, the photo alone gives macro-F1 0.786, the force alone 0.428.
Without force it gives 0.790; without the photo, 0.428. Mean trust is 0.30 for force and 0.49
for the photo, and the mean conflict is 0.07. The model has 1.31 M parameters.

**XJTU-SY** (one bearing per condition held out per fold; normalized RUL; a bag is a
snapshot):

| Run | RMSE | MAE | 90 % coverage | 90 % width | PHM 2012 score |
| --- | --- | --- | --- | --- | --- |
| Mantis F + E (`planb-xjtu-mantis-FE`) | 0.177 ± 0.079 | 0.106 ± 0.068 | 0.79 | 0.215 | 0.662 |
| Handcrafted F + E (`planb-xjtu-stats-FE`) | **0.158 ± 0.073** | 0.093 ± 0.057 | 0.73 | 0.204 | 0.678 |
| Training mean | 0.300 ± 0.073 | 0.240 ± 0.046 | – | – | – |

The PHM 2012 score here averages every snapshot past FPT and before it (where RUL = 1). It
is not the challenge's score at fixed cut-off points, so it compares runs only with each
other.

Reading:
1. **Plan B is a working baseline.** On held-out Nonastreda tools it reaches macro-F1 0.78
   and accuracy 0.82, against a published ~75 % from a less strict protocol. It beats the
   persistence reference on 4 of 5 folds (0.781 against 0.756 on average), but within one
   standard deviation; the paired tests come with the report.
2. **The previous pass's photos carry the prediction.** The photo alone equals the
   combination; force alone is weak (0.43).
3. **Missing-modality robustness is poor.** Without the photo, the combined model falls to
   0.43. A force-only model trained on its own reaches 0.53: checkpoint selection follows the
   combined score, which the photo dominates. This motivates the plan's modality dropout
   (0.3) in the Gate-2 fusion search, and Gate 2 must report the without-photo rows.
4. **Calibration.** The evidential class head is *under*confident (fitted temperature about
   0.4, which sharpens). Temperature scaling lowers the window ECE from 0.24 to 0.11 and the
   Brier score from 0.385 to 0.308.
5. **Regression.** Plan B clearly beats the training mean on wear and RUL, but not
   persistence on wear (23.3 against 20.6 µm). Its 90 % intervals are too narrow: coverage
   0.69 on Nonastreda and 0.73-0.79 on XJTU-SY. See decision D in `docs/USER_ACTIONS.md`.
6. **Encoder.** Mantis and the handcrafted features tie on Nonastreda (0.781 against 0.776),
   and the handcrafted features are better on XJTU-SY RUL (0.158 against 0.177), as in the
   preliminary Gate 1.
7. **Cost.** About 60-70 s per fold and seed on Nonastreda with photos, and 25 s on XJTU-SY,
   with 5 CPU threads.

Two operational notes:
- One fold of the handcrafted Nonastreda run reports 25,000 s. The laptop slept during it;
  its results are unaffected.
- The RAM guard refused three runs at its 2 GB threshold. A run peaks at about 0.6 GB, so the
  threshold is now 1 GB.

### Verification at the end of the phase

| Check | Result |
| --- | --- |
| New tests (synthetic stores and caches; no data needed) | 80: `test_evidential` 23, `test_metrics` 12, `test_inputs` 22, `test_planb` 23; plus one `test_slurm` test (Linux). Line coverage of the new modules 99 % |
| Full suite with the real data and the cached checkpoints | 309 passed, 25 skipped (5 network, 4 Paderborn/PHM 2010 acceptance, 16 Linux-only SLURM tests) |
| CI reproduced in fresh Python 3.11 and 3.12 environments (`scripts/ci/run_local_ci.sh`) | 284 passed, 46 skipped on both; coverage 85 %; pyright 0 errors |
| ruff check and format; pre-commit on every new and changed file | Clean; all hooks pass |
| `slurm/status.sh` Plan B section | Run under Git Bash against a fake run folder (`2 of 15`, summary found); its test runs in CI on Ubuntu |

### Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-10-29 | Plan B experts per channel family, slow channels and photos; context text is not an expert | Above (design decisions) |
| 2026-10-29 | Local Plan B with Mantis F (the preliminary Gate-1 winner) and the handcrafted reference; the final encoder runs on Ruche (§16) | Gate 1 is not final until the GPU caches and Paderborn exist |
| 2026-10-29 | Plan B's RAM guard at 1 GB | Measured peak 0.6 GB; the 2 GB guess refused runs on an idle CPU |
| Open | Calibrated regression intervals (decision D in `docs/USER_ACTIONS.md`) | Coverage 0.69-0.79 for a 90 % interval |
