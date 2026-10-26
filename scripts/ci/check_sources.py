"""Report whether every registered data source is still reachable (never fails the build).

    python scripts/ci/check_sources.py [--markdown summary.md]

Download hosts move and die (PHM 2010's original CDN is now a parking page), so the weekly
``sources`` workflow runs this and posts the table to the job summary. Each check requests
a few bytes or a listing only; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from pdmfm.fetch import USER_AGENT
from pdmfm.sources import SOURCES

CHECKS = {
    "nonastreda": "https://data.mendeley.com/public-api/datasets/m892d2wtzh?version=1",
    "kaist": "https://data.mendeley.com/public-api/datasets/ztmf3m7h5x?version=6",
    "najafi_thermal": "https://data.mendeley.com/public-api/datasets/m4sbt8hbvk?version=3",
    "cwru": "https://engineering.case.edu/sites/default/files/97.mat",
    "paderborn": "https://groups.uni-paderborn.de/kat/BearingDataCenter/K001.rar",
    "xjtu_sy": "https://drive.google.com/drive/folders/1_ycmG46PARiykt82ShfnFfyQsaXv3_VK",
    "phm2010": "http://cdn.teamholistic.net/phmconference.org/c1.zip",
    "mbhm": "https://huggingface.co/api/datasets/SIA-IDE/MBHM/tree/main",
    "nasa_milling": "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip",
}


def check(url: str) -> tuple[str, str]:
    """(status, detail) for one URL: OK when it answers with data rather than an HTML page."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-255"},
            timeout=30,
            stream=True,
            allow_redirects=True,
        )
    except requests.RequestException as error:
        return "UNREACHABLE", type(error).__name__
    content_type = response.headers.get("Content-Type", "")
    if response.status_code >= 400:
        return "ERROR", f"HTTP {response.status_code}"
    if "text/html" in content_type and "drive.google.com" not in url:
        return "SUSPECT", f"HTML page instead of data ({response.url[:60]})"
    return "OK", f"HTTP {response.status_code} {content_type.split(';')[0]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", type=Path, help="also write a markdown table here")
    args = parser.parse_args()
    rows = []
    for name, url in CHECKS.items():
        status, detail = check(url)
        rows.append((name, SOURCES[name].role, status, detail))
        print(f"{name:15s} {status:12s} {detail}")
    if args.markdown:
        lines = ["| Dataset | Role | Status | Detail |", "| --- | --- | --- | --- |"]
        lines += [f"| {n} | {r} | {s} | {d} |" for n, r, s, d in rows]
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
