# Continuous integration

Normal pushes/PRs run lint and formatting, Linux Python 3.11 core coverage, Linux 3.12
compatibility, and Windows 3.11 numerical/reader/window/end-to-end contracts. Coverage is
collected once. Type checking remains advisory while third-party typing is incomplete.
No private checkout or repository token is needed.

The weekly/manual quality job runs the full non-acceptance suite including multi-seed
learning checks on Linux/Windows and Python 3.11/3.12 with optional encoder dependencies.
Real checkpoints, datasets, network sources and CUDA are separate opt-in acceptance jobs.
The sources workflow checks publisher availability weekly. Cluster `checks.sbatch` opts
into available real-data checks; GPU checks use the commands in TESTING.md.

Local equivalent: `bash scripts/ci/run_local_ci.sh`. See [test categories](TESTING.md).
Historical CI notes remain in [archive](archive/CI-previous.md).
