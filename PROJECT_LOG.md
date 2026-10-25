# Project 14 — project log

This is the running record of tasks, progress, findings, implementations and decisions. Each week has its own section, newest last. The plan it follows is the [Project 14 implementation plan](https://claude.ai/artifact/1M2DPuffJP1mRt2oiUujx5); a reference copy is in [`docs/PLAN.md`](docs/PLAN.md).

## Status at a glance (updated 2026-10-25)

| Area | State |
| --- | --- |
| Phase | Week 1 (planned Oct 26 to Nov 1; started Oct 25) |
| Repository, environment, tests | Done: package skeleton, downloader, 14 offline tests, lint clean |
| Datasets on `D:\Project14\data` | Done: CWRU, Najafi thermal, MBHM metadata, Nonastreda (your local copy reused, chip and workpiece photos added). XJTU-SY (extracted, verified), KAIST. **Blocked:** Paderborn (host refuses connections), PHM 2010 (original host gone) |
| Encoders | All 7 load and run on CPU (Mantis, MOMENT-base, DINOv2-S, Qwen3-Embedding, AST, TimesFM 3.0, TimesFM 2.5) |
| Week-1 checklist | 10 of 12 settled; the 2 open items need your input (PHM 2010 access, Ruche network test) |
| Needs your input | PHM 2010 access, Paderborn route, Ruche network test (see "Needs your input") |

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
