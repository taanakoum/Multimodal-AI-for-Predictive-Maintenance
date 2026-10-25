"""Week-1 check: CPU throughput of the frozen encoders, to size the feature-caching stage.

    python scripts/week1/encoder_throughput.py

Measures windows (or images) per second at realistic batch sizes after a warm-up, using all
CPU threads, and prints the time needed for the plan's corpus estimate.
"""

from __future__ import annotations

import json
import os
import time

import torch

from pdmfm.paths import data_root

torch.set_num_threads(os.cpu_count() or 4)
# Plan estimate: ~113k channel-windows; the T arm encodes 24 segments of 512 per window.
SEGMENTS_T = 113_000 * 24
SPECTRA_FE = 113_000 * 2
IMAGES = 378 * 3 + 409  # Nonastreda tool, chip, workpiece photos + Najafi thermal


def rate(fn, batch, repeats=3) -> float:
    fn(batch)  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn(batch)
    return repeats * len(batch) / (time.perf_counter() - start)


def main() -> None:
    results = {"threads": torch.get_num_threads()}
    from mantis.architecture import MantisV1

    mantis = MantisV1(device="cpu").from_pretrained("paris-noah/Mantis-8M").eval()
    x = torch.randn(256, 1, 512)
    with torch.no_grad():
        results["mantis_per_s"] = rate(lambda b: mantis(b), x)

    from momentfm import MOMENTPipeline

    moment = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base", model_kwargs={"task_name": "embedding"}
    )
    moment.init()
    moment.eval()
    with torch.no_grad():
        results["moment_base_per_s"] = rate(lambda b: moment(x_enc=b), x[:64])

    from transformers import AutoModel

    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    images = torch.randn(8, 3, 224, 672)
    with torch.no_grad():
        results["dinov2_224x672_per_s"] = rate(lambda b: dino(pixel_values=b), images, 2)

    for name in ("mantis", "moment_base"):
        per_s = results[f"{name}_per_s"]
        results[f"{name}_T_arm_hours"] = SEGMENTS_T / per_s / 3600
        results[f"{name}_F_E_arms_hours"] = SPECTRA_FE / per_s / 3600
    results["dinov2_all_images_minutes"] = IMAGES / results["dinov2_224x672_per_s"] / 60
    out = data_root().parent / "logs" / "encoder_throughput.json"
    out.write_text(json.dumps(results, indent=2))
    for key, value in results.items():
        print(f"{key:28s} {value:10.2f}" if isinstance(value, float) else f"{key:28s} {value}")


if __name__ == "__main__":
    main()
