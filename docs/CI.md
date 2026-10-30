# Tests and continuous integration

Three layers run the same checks:

1. the local test suite;
2. pre-commit hooks before each commit;
3. GitHub Actions on every push.

CI never needs datasets, model weights or a GPU. Tests that read real data skip themselves,
and they run on the development machine (and on Ruche) instead.

## 1. Test suite

```powershell
. .\scripts\env\dev.ps1
pytest                                   # everything available (real-data tests run when data is on disk)
pytest -m "not data"                     # what CI runs: synthetic data only
pytest -m data -rs                       # only the tests that read the real downloads
$env:PDMFM_NETWORK_TESTS = "1"; pytest -m network   # publishers still serve the files we used
```

| Marker | Meaning | When it runs |
|---|---|---|
| (none) | Unit and integration tests on synthetic files written in each publisher's format | Always, including CI |
| `data(name, extracted=…)` | Reads the real dataset under `PDMFM_DATA_ROOT` | Only when that dataset is on disk; skipped with a reason otherwise |
| `network` | Contacts the publishers (sizes, hashes, metadata) | Only with `PDMFM_NETWORK_TESTS=1` (weekly workflow) |
| `slow` | Over about 10 s (loads all of KAIST, Nonastreda) | Always where its data exists; deselect with `-m "not slow"` |
| `weights(repo)` | Loads a pretrained checkpoint (Mantis, MOMENT, AST, DINOv2, Qwen3) | Only when it is in the Hugging Face cache; never in CI. On Ruche, step 0 runs them offline |

| File | Covers |
|---|---|
| `test_paths.py`, `test_sources.py` | Data-root resolution; the dataset registry (unique paths, licences, Paderborn split lists) |
| `test_fetch.py` | Downloader: hashes, resume, HTML rejection, manifests, seeding from a local copy, parallel downloads, Mendeley paths, archive extraction, `verify` for hand-placed files |
| `test_schema.py` | The as-of join never attaches an image captured at or after the window start; missing kinds stay absent |
| `test_preprocess.py` | Exact resampling ratios and anti-aliasing; windows; F- and E-arm spectra; the 5 kHz band limit (flat passband, ≥ 80 dB stop band, no delay, recorder roll-off case); dominant-frequency estimate; level descriptors; FPT; RUL |
| `test_resources.py` | CPU/RAM guard (busy refusal, waiting, SLURM cores, thread caps) |
| `test_readers.py` | Every reader on synthetic MAT/TDMS/CSV/BMP files in the real layout (Nonastreda through a stand-in `omnifuser.data`) |
| `test_eda.py` | EDA helpers; CWRU and Najafi analyses end to end on synthetic files |
| `test_texts.py` | Context and fault-description YAMLs: every reader label has a class; contexts never name the state; descriptions never name speed or load |
| `test_protocols.py` | Unit-disjoint outer folds; every unit tested once; Paderborn stratification; pretraining never sees test, validation or target units; few-shot and subsample draws |
| `test_windows.py` | Window stores on synthetic readers: exact model input, labels, load-only contexts, measured speed, XJTU-SY RUL labels, previous-pass photos only, slow channels, atomic rebuilds; the T/F/E/S arms |
| `test_encoders.py` | Handcrafted encoder; registry; AST position resizing and PatchTST encoding on tiny models; the real checkpoints (`weights`) |
| `test_features.py` | Cached features equal online encoding; resume after a crash; subsamples; signature changes give new folders; photo and text caches |
| `test_probe.py` | Gate 1 on synthetic stores: scores per task, validation units only, shared rows, per-fold models, inapplicable arms ranked last, tie to the cheaper encoder, parallel = serial |
| `test_patchtst.py` | Pretraining corpus per fold; resume after an interruption; refusal of other settings; the checkpoint loads as an encoder |
| `test_evidential.py` | Dempster's rule on hand-computed opinions (conflict, agreement, vacuous identity, commutativity, associativity); TMC loss and KL by closed form; NIG NLL equals the Student-t log density; MoNIG by hand; absent modalities skipped; a fitted NIG gives 90 % coverage |
| `test_metrics.py` | ECE, Brier and NLL by hand; macro-F1 and balanced accuracy over the classes present; temperature scaling repairs a known overconfidence and leaves calibrated predictions alone; Student-t NLL, interval coverage and width; the interval scale recovers a known factor and gives 90 % coverage; PHM 2012 score values |
| `test_inputs.py` | Token sets from the caches: cached tokens copied unchanged, roles and channels, no Fz level, window means, level modes (absolute, relative, both; the force mean descriptor), levels relative to each tool's first pass (and, for Paderborn, to the healthy *training* bearings per setting), KAIST refused until stage 8, previous-pass photos with age and masks, slow-channel tokens, absent rows for uncached windows, text prototypes in YAML order, and scrambled `meta_*` manifest columns that change no input |
| `test_planb.py` | Plan B: shapes; an absent or dropped modality leaves the other opinion and NIG unchanged; masked tokens have no effect; ≤ 3 M parameters at full size; reproducible training; learning on synthetic wear classes; resume and refusal of other settings; persistence baselines on a known sequence; level and slow descriptors standardized with training rows only; calibrated intervals equal the raw ones times the fitted factor; per-fold inputs (Paderborn, `{fold}` encoders); `--without`; the command line |
| `test_slurm.py` | The Ruche scripts parse; chains link each part to the previous one; jobs refuse to run outside SLURM; status runs anywhere and counts Plan B results (Linux only) |
| `test_real_data.py` | Facts verified on the real downloads and the window stores built from them, kept as regression tests; acceptance tests for Paderborn and PHM 2010 (`docs/USER_ACTIONS.md`) |
| `test_network.py` | Publisher sizes and hashes (Najafi, KAIST, MBHM, CWRU, NASA) |

## 2. pre-commit

`.pre-commit-config.yaml` runs, on the files of each commit:

- YAML/TOML syntax;
- end-of-file and trailing-whitespace fixes;
- merge-conflict markers;
- a 1 MB file limit, so data stays out of git;
- ruff lint and format;
- `actionlint` on the workflows.

Install once with `pre-commit install`. Ruff is pinned to **0.17.2** in three places that
must agree: `pyproject.toml` (dev extra), `.pre-commit-config.yaml`, and the CI lint job.
Formatting changes between ruff versions, so a mismatch makes CI reject code that passes
locally. Dependabot proposes bumps; update all three together.

## 3. GitHub Actions

### `CI` (`.github/workflows/tests.yml`): pushes to `master`, every pull request, or by hand

Pushes to other branches run through their pull request, so each commit runs once, not twice.
To test a branch without a pull request: **Actions → CI → Run workflow**.

| Job | Runs on | What it does | Blocking |
|---|---|---|---|
| Lint and format | Ubuntu, Python 3.11 | `ruff check`, `ruff format --check`, `actionlint` (with shellcheck on `run:` blocks) | Yes |
| Tests (4 jobs) | Ubuntu and Windows × Python 3.11 and 3.12 | Installs CPU-only torch, then `.[dev,eda]`; runs the entry points (`pdmfm.fetch list`, `pdmfm.resources`) and pytest with coverage; uploads `junit.xml` and `coverage.xml` | Yes; `fail-fast: false`, so every leg reports |
| Type check | Ubuntu | `pyright pdmfm` (0 errors since 2026-10-26) | No (`continue-on-error`); a red result does not fail the run |

Design choices:
- **CPU torch from the PyTorch index.** The default PyPI wheel for Linux bundles CUDA
  libraries (about 2 GB); the CPU wheel is about 200 MB.
- **Empty data root.** `PDMFM_DATA_ROOT` points at an empty folder, so every `data` test skips
  with a reason instead of failing.
- **Windows leg.** The development machine is Windows, and paths, encodings and RAR tools
  differ there.
- **Cancel superseded runs.** A newer push to the same branch cancels a running one.
- **Read-only permissions.** Workflows get `contents: read` only.
- **Optional OmniFuser secret.** With `OMNIFUSER_READ_TOKEN` set, CI also installs the private
  `omnifuser` package. No current test needs it.

### `Data sources` (`.github/workflows/sources.yml`): Mondays 05:17 UTC, or by hand

1. A reachability table for every download host, posted to the job summary. It never fails:
   hosts go down.
2. `pytest -m network`, which **fails** when a publisher has changed a file's size or hash.
   A dataset that changes silently would break reproducibility.

### Dependabot (`.github/dependabot.yml`)

Weekly grouped pull requests for GitHub Actions versions and Python dependencies.

## 4. Reliability check (done locally; last on 2026-10-29)

| Check | How | Result |
|---|---|---|
| Workflow syntax and expressions | `actionlint` 1.7.12 on both workflows | No findings |
| CI reproduced, Python 3.11 | `bash scripts/ci/run_local_ci.sh 3.11`: fresh venv in a temp folder, CPU torch, `.[dev,eda]`, no OmniFuser, empty data root, then the CI steps | **104 passed, 18 skipped** (5 network, 13 data); coverage 91 %; 108 s with a warm uv cache |
| CI reproduced, Python 3.12 | `bash scripts/ci/run_local_ci.sh 3.12` (uv downloaded a managed 3.12) | **104 passed, 18 skipped**; coverage 91 %; 171 s |
| Type check reproduced | `run_local_ci.sh` now ends with `pyright pdmfm`. Before the fixes, pyright against the project environment reported 9 errors: three reader robustness bugs and false alarms from the private `omnifuser` import, scipy stubs and `__doc__` under `-OO`. A CI-like environment also lacked `gdown`, a missing dependency. After the fixes | **0 errors**; 107 passed in a fresh 3.11 environment |
| Hooks | `pre-commit run` on every tracked and untracked file | All 9 hooks pass, after two fixes it found (EDA JSON without a final newline; the renamed `ruff-check` hook id) |
| Full local suite with real data | `pytest` with `PDMFM_DATA_ROOT=D:\Project14\data` | **111 passed, 5 skipped** (network) before the acceptance and regression tests were added; those then passed (2) or skipped (4, data absent) |
| Acceptance tests themselves | Run against a synthetic layout built to the Paderborn and PHM 2010 documentation | 4 passed: the test code works before the real files arrive |
| Weeks 2-4, fresh environments (2026-10-29) | `run_local_ci.sh` on Python 3.11 and 3.12, now ending with pyright | **204 passed, 45 skipped** on both; pyright **0 errors**; coverage 80 % (the encoder wrappers need the optional libraries, so CI cannot run them; they are tested locally against the real checkpoints) |
| Weeks 2-4, full local suite | `pytest` with the real data and cached checkpoints | **229 passed, 24 skipped** (network, Paderborn and PHM 2010 acceptance, Linux-only SLURM tests) |
| What the fresh environment caught | pandas 3.1 (installed by CI) types text columns as strings; one test asserted `object` | Test made version-independent; it would have failed on GitHub |
| Week 5 (Plan B), fresh environments (2026-10-29) | `run_local_ci.sh` on Python 3.11 and 3.12 | **284 passed, 46 skipped** on both; pyright **0 errors**; coverage 85 %; the new modules (losses, metrics, model, train, planb) at 99 % |
| Week 5, full local suite | `pytest` with the real data and cached checkpoints | **309 passed, 25 skipped** (5 network, 4 Paderborn and PHM 2010 acceptance, 16 Linux-only SLURM tests) |
| Week 5 after the audit (2026-10-30) | Full local suite; `run_local_ci.sh` on 3.11 and 3.12 | **315 passed, 25 skipped** locally; **290 passed, 46 skipped** in both fresh environments; coverage 85 %; pyright **0 errors** |

**Not covered by CI:**
- the real data (local and Ruche only);
- GPU code paths;
- the encoder extras (`mantis-tsfm`, `transformers` models): the weights are too large for CI;
  the `weights` tests in `test_encoders.py` load the real checkpoints locally and in the Ruche
  step-0 job;
- the Ruche shell scripts, beyond `bash -n` syntax checks.

**Run CI locally before pushing:** `bash scripts/ci/run_local_ci.sh` (Git Bash, needs `uv`).

## 5. When a job is red

| Symptom | Likely cause | Fix |
|---|---|---|
| Lint: `ruff format --check` lists files | Code not formatted with ruff 0.17.2 | `ruff format pdmfm tests scripts` (or let pre-commit do it) |
| Tests fail only on Windows | Path separators, file encodings (`PYTHONIOENCODING`), file locking | Reproduce locally; the dev machine is Windows |
| Tests fail only on 3.12 | A dependency without 3.12 wheels, or a deprecation | Run `scripts/ci/run_local_ci.sh 3.12` |
| A test passes locally but fails in CI | It needs data or network but lacks the marker | Add `@pytest.mark.data(...)` or `network` |
| `Data sources`: network test fails | A publisher changed or removed a file | Compare with `MANIFEST.json`; record the new hash only after checking what changed |
| Out of Actions minutes | Windows minutes count double on private repositories | Run the Windows legs on pull requests only (below) |

To run the Windows legs on pull requests only, replace the test matrix's `os` line with the
following (checked with actionlint):

```yaml
        os: ${{ github.event_name == 'pull_request' && fromJSON('["ubuntu-latest", "windows-latest"]') || fromJSON('["ubuntu-latest"]') }}
```
