"""Encoder wrappers (pdmfm.encoders). Real checkpoints only under the ``weights`` marker."""

from __future__ import annotations

import numpy as np
import pytest

from pdmfm import encoders
from pdmfm import preprocess as pp


def test_stats_encoder_band_means_maxima_and_moments():
    x = np.tile(np.arange(512, dtype=np.float32), (3, 1))
    z = encoders.StatsEncoder().encode(x)
    assert z.shape == (3, 36) and z.dtype == np.float32
    np.testing.assert_allclose(z[0, :16], np.arange(512).reshape(16, 32).mean(axis=1))
    np.testing.assert_allclose(z[0, 16:32], np.arange(31, 512, 32))
    np.testing.assert_allclose(z[0, 32:], [x[0].mean(), x[0].std(), 0, 511], rtol=1e-6)
    assert encoders.StatsEncoder().signature() == {"encoder": "stats", "dim": 36, "version": 2}


def test_stats_encoder_keeps_a_narrow_line():
    floor = np.full((2, 512), -2.4, dtype=np.float32)  # a flat log spectrum
    line = floor.copy()
    line[:, 40] = 0.0  # one bin two decades above the floor, as a defect line
    plain, tone = encoders.StatsEncoder().encode(floor), encoders.StatsEncoder().encode(line)
    assert tone[0, 1] - plain[0, 1] < 0.1  # the band mean barely moves ...
    assert tone[0, 17] - plain[0, 17] > 2.0  # ... the band maximum shows the line


def test_registry_and_costs():
    assert isinstance(encoders.get_encoder("stats"), encoders.StatsEncoder)
    with pytest.raises(KeyError, match="unknown encoder"):
        encoders.get_encoder("no-such-encoder")
    with pytest.raises(ValueError, match="checkpoint"):
        encoders.get_encoder("patchtst")
    assert encoders.cost("mantis") < encoders.cost("patchtst-fold3") < encoders.cost("moment")
    assert encoders.cost("unknown") > encoders.cost("ast")


def test_hf_revision_of_an_uncached_repo_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert encoders.hf_revision("nobody/not-a-repo") is None


def test_cuda_request_without_cuda_is_refused():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(RuntimeError, match="CUDA"):
        encoders._torch_device("cuda")


def test_ast_positions_resize_for_short_inputs():
    pytest.importorskip("transformers")
    import torch
    from transformers import ASTConfig, ASTModel

    torch.manual_seed(0)
    config = ASTConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        max_length=128, num_mel_bins=32,
    )  # fmt: skip
    model = ASTModel(config).eval()
    short = encoders.resize_ast_positions(model, 48)
    f, t = short.embeddings.get_shape(short.config)
    assert short.embeddings.position_embeddings.shape == (1, 2 + f * t, 32)
    # Everything except the position grid is copied unchanged.
    for name, value in model.state_dict().items():
        if name != "embeddings.position_embeddings":
            torch.testing.assert_close(short.state_dict()[name], value)
    # The two special tokens keep their embeddings; the grid is interpolated, not reset.
    torch.testing.assert_close(
        short.embeddings.position_embeddings[:, :2], model.embeddings.position_embeddings[:, :2]
    )
    with torch.no_grad():
        out = short(input_values=torch.randn(2, 48, 32)).pooler_output
    assert out.shape == (2, 32) and torch.isfinite(out).all()


@pytest.fixture
def tiny_patchtst(tmp_path):
    pytest.importorskip("transformers")
    from transformers import PatchTSTConfig, PatchTSTForPretraining

    config = PatchTSTConfig(
        num_input_channels=1, context_length=512, patch_length=16, patch_stride=16, d_model=16,
        num_hidden_layers=1, num_attention_heads=2, ffn_dim=32, do_mask_input=True,
        mask_type="random", random_mask_ratio=0.4, scaling="std",
    )  # fmt: skip
    folder = tmp_path / "fold1" / "model"
    PatchTSTForPretraining(config).save_pretrained(folder)
    return folder


def test_patchtst_encoder_reads_a_pretraining_checkpoint(tiny_patchtst):
    encoder = encoders.get_encoder("patchtst", checkpoint=tiny_patchtst)
    assert encoder.name == "patchtst-fold1" and encoder.dim == 16
    x = np.random.default_rng(0).standard_normal((5, 512)).astype(np.float32)
    z = encoder.encode(x)
    assert z.shape == (5, 16)
    np.testing.assert_array_equal(z, encoder.encode(x))  # no masking when encoding
    digest = encoder.signature()["checkpoint"]
    assert len(digest) == 16


# ----------------------------------------------------------------------------- real weights

SERIES = np.random.default_rng(1).standard_normal((3, 512)).astype(np.float32)


@pytest.mark.slow
@pytest.mark.weights("paris-noah/Mantis-8M")
def test_mantis_checkpoint():
    encoder = encoders.get_encoder("mantis")
    z = encoder.encode(SERIES)
    assert z.shape == (3, 256) and np.isfinite(z).all()
    assert encoder.signature()["revision"]


@pytest.mark.slow
@pytest.mark.weights("AutonLab/MOMENT-1-base")
def test_moment_checkpoint():
    pytest.importorskip("momentfm")
    z = encoders.get_encoder("moment").encode(SERIES)
    assert z.shape == (3, 768) and np.isfinite(z).all()


@pytest.mark.slow
@pytest.mark.weights("MIT/ast-finetuned-audioset-10-10-0.4593")
def test_ast_checkpoint_on_one_second_windows():
    encoder = encoders.get_encoder("ast")
    window = np.random.default_rng(2).standard_normal((2, pp.WINDOW_SAMPLES)).astype(np.float32)
    z = encoder.encode(window)
    assert z.shape == (2, 768) and np.isfinite(z).all()
    assert encoder.signature()["max_frames"] == 100


@pytest.mark.slow
@pytest.mark.weights("facebook/dinov2-small")
def test_dinov2_photo_tokens(tmp_path):
    from PIL import Image

    path = tmp_path / "tool.png"
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (500, 1550, 3), dtype=np.uint8)).save(
        path
    )
    encoder = encoders.DinoV2Encoder()
    tokens = encoder.encode_paths([path, path], "tool")
    assert tokens.shape == (2, 9, 384) and encoder.n_tokens("chip") == 5
    np.testing.assert_allclose(tokens[0], tokens[1], rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.weights("Qwen/Qwen3-Embedding-0.6B")
def test_qwen3_text_embeddings_are_normalized():
    z = encoders.Qwen3TextEncoder().encode(["An outer-race defect.", "A healthy bearing."])
    assert z.shape == (2, 1024)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, rtol=1e-4)
