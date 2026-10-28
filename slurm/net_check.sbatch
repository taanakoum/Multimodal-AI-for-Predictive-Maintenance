#!/bin/bash
# Can Ruche reach every host this project downloads from? Run it twice and compare:
#   compute node:  sbatch slurm/net_check.sbatch   -> runs/logs/net-check-<jobid>.out
#   login node:    bash slurm/net_check.sbatch     -> printed here
# One OK/FAIL line per host (HTTP code; 000 = no connection), then a VERDICT line.
# How to read the result: docs/RUCHE_RUNBOOK.md, section 12.
#SBATCH --job-name=net-check
#SBATCH --output=runs/logs/%x-%j.out
#SBATCH --partition=cpu_short
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

echo "node: $(hostname)  date: $(date -Is)  slurm job: ${SLURM_JOB_ID:-none (login node)}"
echo "proxy: http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"

hosts=(
  "weights      https://huggingface.co/api/models/facebook/dinov2-small"
  "pip          https://pypi.org/simple/torch/"
  "torch-cuda   https://download.pytorch.org/whl/cu126/"
  "github       https://github.com/"
  "mendeley     https://data.mendeley.com/public-api/datasets/ztmf3m7h5x?version=6"
  "paderborn    https://groups.uni-paderborn.de/kat/BearingDataCenter/K001.rar"
  "cwru         https://engineering.case.edu/sites/default/files/97.mat"
  "gdrive       https://drive.google.com/"
  "nasa-s3      https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
)
ok=0
failed=()
for entry in "${hosts[@]}"; do
  read -r name url <<<"$entry"
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -r 0-0 "$url")
  if [[ "$code" =~ ^(200|206|301|302|303|307|308)$ ]]; then
    echo "OK   $code $name $url"
    ok=$((ok + 1))
  else
    echo "FAIL $code $name $url"
    failed+=("$name")
  fi
done

total=${#hosts[@]}
if ((ok == total)); then
  echo "VERDICT: all $total hosts reachable from $(hostname)"
elif ((ok == 0)); then
  echo "VERDICT: no internet on $(hostname); pre-download on the login node (HF_HUB_OFFLINE=1 in jobs)"
else
  echo "VERDICT: $ok of $total reachable from $(hostname); failing: ${failed[*]}"
fi
