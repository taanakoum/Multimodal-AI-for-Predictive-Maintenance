"""Week-1 check: every pretrained encoder in the plan loads and runs on CPU.

For each encoder this records the checkpoint, parameter count, input it was given, output
shape, load time and forward time, and writes everything to
``$PDMFM_DATA_ROOT/../logs/encoder_smoke.json``. One failure does not stop the others.

    python scripts/week1/encoder_smoke.py [--only mantis,dinov2]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

from pdmfm.paths import data_root

torch.manual_seed(0)
RNG = np.random.default_rng(0)


def _params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def mantis() -> dict:
    from mantis.architecture import MantisV1
    from mantis.trainer import MantisTrainer

    network = MantisV1(device="cpu").from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device="cpu", network=network)
    x = RNG.standard_normal((4, 1, 512)).astype(np.float32)  # 4 univariate 512-point windows
    start = time.perf_counter()
    z = model.transform(x)
    return {
        "checkpoint": "paris-noah/Mantis-8M",
        "params_m": _params(network) / 1e6,
        "input": list(x.shape),
        "output": list(np.asarray(z).shape),
        "forward_s": time.perf_counter() - start,
    }


def moment() -> dict:
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base", model_kwargs={"task_name": "embedding"}
    )
    model.init()
    model.eval()
    x = torch.randn(4, 1, 512)
    start = time.perf_counter()
    with torch.no_grad():
        out = model(x_enc=x)
    return {
        "checkpoint": "AutonLab/MOMENT-1-base",
        "params_m": _params(model) / 1e6,
        "input": list(x.shape),
        "output": list(out.embeddings.shape),
        "forward_s": time.perf_counter() - start,
    }


def dinov2() -> dict:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    x = torch.randn(1, 3, 224, 672)  # Nonastreda photo resized to 16 x 48 patches
    start = time.perf_counter()
    with torch.no_grad():
        out = model(pixel_values=x)
    return {
        "checkpoint": "facebook/dinov2-small",
        "params_m": _params(model) / 1e6,
        "input": list(x.shape),
        "output": list(out.last_hidden_state.shape),
        "forward_s": time.perf_counter() - start,
    }


def qwen3_embedding() -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    texts = [
        "Outer-race defect of 1.0 mm on a deep-groove ball bearing.",
        "Inner-race defect of 1.0 mm on a deep-groove ball bearing.",
        "Healthy deep-groove ball bearing with no defect.",
    ]
    start = time.perf_counter()
    z = model.encode(texts, normalize_embeddings=True)
    sims = (z @ z.T).round(3).tolist()
    return {
        "checkpoint": "Qwen/Qwen3-Embedding-0.6B",
        "params_m": _params(model) / 1e6,
        "input": f"{len(texts)} sentences",
        "output": list(z.shape),
        "forward_s": time.perf_counter() - start,
        "cosine_outer_inner_healthy": sims,
    }


def ast() -> dict:
    from transformers import ASTFeatureExtractor, ASTModel

    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    extractor = ASTFeatureExtractor.from_pretrained(name)
    model = ASTModel.from_pretrained(name).eval()
    wave = RNG.standard_normal(16_384).astype(np.float32)  # 1.024 s at 16 kHz
    features = extractor(wave, sampling_rate=16_000, return_tensors="pt")
    start = time.perf_counter()
    with torch.no_grad():
        out = model(**features)
    return {
        "checkpoint": name,
        "params_m": _params(model) / 1e6,
        "input": list(features["input_values"].shape),
        "output": list(out.pooler_output.shape),
        "forward_s": time.perf_counter() - start,
        "extractor": {
            "sampling_rate": extractor.sampling_rate,
            "num_mel_bins": extractor.num_mel_bins,
            "max_length": extractor.max_length,
        },
    }


def timesfm3() -> dict:
    from timesfm import TimesFM3Forecaster

    model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
    context = np.sin(np.arange(512) / 10.0) + 0.05 * RNG.standard_normal(512)
    start = time.perf_counter()
    out = model.predict(context, horizon=32, return_quantiles=True)
    point, quantiles = out.forecast, out.quantiles
    return {
        "checkpoint": "google/timesfm-3.0-pytorch (non-commercial weights)",
        "params_m": _params(model.model) / 1e6,
        "input": "context 512, horizon 32",
        "output": {
            "point": list(np.asarray(point).shape) if point is not None else str(type(out)),
            "quantiles": list(np.asarray(quantiles).shape) if quantiles is not None else None,
        },
        "forward_s": time.perf_counter() - start,
    }


def timesfm25() -> dict:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128))
    context = np.sin(np.arange(512) / 10.0)
    start = time.perf_counter()
    point, quantiles = model.forecast(horizon=32, inputs=[context])
    return {
        "checkpoint": "google/timesfm-2.5-200m-pytorch (Apache-2.0)",
        "params_m": _params(model.model) / 1e6,
        "input": "context 512, horizon 32",
        "output": {"point": list(point.shape), "quantiles": list(quantiles.shape)},
        "forward_s": time.perf_counter() - start,
    }


ENCODERS = {
    "mantis": mantis,
    "moment": moment,
    "dinov2": dinov2,
    "qwen3_embedding": qwen3_embedding,
    "ast": ast,
    "timesfm3": timesfm3,
    "timesfm25": timesfm25,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated subset of: " + ", ".join(ENCODERS))
    args = parser.parse_args()
    names = args.only.split(",") if args.only else list(ENCODERS)
    out_path = data_root().parent / "logs" / "encoder_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    print(f"HF_HOME={os.environ.get('HF_HOME')}  torch={torch.__version__}")
    for name in names:
        start = time.perf_counter()
        try:
            record = ENCODERS[name]()
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - record every failure and continue
            record = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}"[:800],
                "trace": traceback.format_exc()[-1500:],
            }
        record["total_s"] = round(time.perf_counter() - start, 1)
        results[name] = record
        print(f"{name:16s} {record['status']:7s} {record.get('output', record.get('error'))}")
        out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
