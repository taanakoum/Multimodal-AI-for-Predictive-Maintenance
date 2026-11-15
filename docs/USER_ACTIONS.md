# External access and pending validation

No private repository credentials are required for this project.

1. Paderborn and PHM 2010 real-file acceptance remains pending because their raw datasets
   are absent from the local data root. Use `python -m pdmfm.fetch plan <dataset>` to inspect
   current requirements. The current Paderborn dataset page is linked in DATASET_AUDIT.md.
2. On Ruche, run CUDA acceptance and `python -m pdmfm.benchmark` against cached inputs before
   launching the full experiment manifest. The current development environment has CPU-only
   PyTorch; no GPU speedup or CUDA pass is claimed from local checks.
3. Source evaluation, target adaptation and final publication are separate research stages.
   See PLAN.md for what remains unimplemented. No final result table is implied by a passing
   synthetic demo.

The previous machine-specific checklist is preserved in archive/USER_ACTIONS-previous.md.
