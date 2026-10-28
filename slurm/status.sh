#!/bin/bash
# One-screen overview: your jobs, the window stores, the feature caches, the PatchTST
# pretraining runs, the Gate-1 runs, logs that contain errors, and disk use.
# Usage, from anywhere:  source slurm/env.sh && bash slurm/status.sh
# It only reads files, so it may run on the login node.
cd "$(dirname "$0")/.." || exit 1

echo "== Your jobs (ST: PD waiting, R running)"
if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o "%.10i %.30j %.2t %.10M %.11l %R"
else
    echo "  (squeue is not available here)"
fi

if ! python -c "import pdmfm" >/dev/null 2>&1; then
    echo
    echo "(load the environment first for the rest: source slurm/env.sh)"
    exit 0
fi

echo
echo "== Window stores"
python -m pdmfm.windows info | sed 's/^/  /'
echo
echo "== Feature caches"
python -m pdmfm.features status | sed 's/^/  /'

echo
echo "== PatchTST pretraining"
found=0
for progress in runs/patchtst/*/progress.json; do
    [ -f "$progress" ] || continue
    found=1
    python - "$progress" <<'EOF'
import json, os, sys
p = json.load(open(sys.argv[1]))
run = os.path.basename(os.path.dirname(sys.argv[1]))
done = "complete" if p["epoch"] >= p["epochs"] else "in progress"
print(f"  {run:10s} {done:12s} epoch {p['epoch']}/{p['epochs']}  best held-out loss {p['best_held_out']:.4f}")
EOF
done
[ "$found" -eq 1 ] || echo "  none started"

echo
echo "== Gate-1 runs"
found=0
for report in runs/gate1/*/report.json; do
    [ -f "$report" ] || continue
    found=1
    python -c "import json, sys; r = json.load(open(sys.argv[1])); print(f\"  {r['run_id']:24s} winner {r['winner']}  ({len(r['skipped'])} skipped)\")" "$report"
done
[ "$found" -eq 1 ] || echo "  none yet"

echo
echo "== Logs with an error (time-limit stops are expected and not listed; read a log's end with: tail -n 40 FILE)"
found=0
for log in runs/logs/*.out; do
    [ -f "$log" ] || continue
    last=$(grep -E "Traceback|Error:|error:|CANCELLED|oom-kill" "$log" | grep -v "DUE TO TIME LIMIT" | tail -n 1)
    if [ -n "$last" ]; then
        echo "  $log: $(echo "$last" | cut -c1-110)"
        found=1
    fi
done
[ "$found" -eq 1 ] || echo "  none"

echo
echo "== Disk use: runs $(du -sh runs 2>/dev/null | cut -f1), data $(du -sh "${PDMFM_DATA_ROOT:-data}" 2>/dev/null | cut -f1)"
