#!/bin/bash
# Week-1 check: can Ruche compute nodes reach the internet?
# Submit from the repository root:  sbatch scripts/ruche/net_check.sbatch
# Result: logs/net_check-<jobid>.out  (one line per host: OK / FAIL)
#SBATCH --job-name=pdmfm-netcheck
#SBATCH --partition=cpu_short
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/net_check-%j.out

echo "node: $(hostname)  date: $(date -Is)"
for url in \
  https://huggingface.co/api/models/facebook/dinov2-small \
  https://data.mendeley.com/public-api/datasets/ztmf3m7h5x?version=6 \
  https://groups.uni-paderborn.de/kat/BearingDataCenter/ \
  https://engineering.case.edu/sites/default/files/97.mat \
  https://pypi.org/simple/torch/ ; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 -r 0-0 "$url")
  if [[ "$code" =~ ^(200|206|301|302)$ ]]; then echo "OK   $code $url"; else echo "FAIL $code $url"; fi
done
