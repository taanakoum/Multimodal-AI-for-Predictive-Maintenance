"""Own PatchTST pretraining (pdmfm.patchtst) at toy size on CPU."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import cwru_reader, xjtu_reader

from pdmfm import protocols, windows

pytest.importorskip("transformers")
from pdmfm import patchtst  # noqa: E402
from pdmfm.encoders import get_encoder  # noqa: E402

TINY = {"d_model": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "ffn_dim": 32}


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("patchtst")
    windows.build("cwru", out=root / "cwru", reader=cwru_reader(), log=quiet)
    windows.build("xjtu_sy", out=root / "xjtu_sy", reader=xjtu_reader(20), log=quiet)
    return root


def test_corpus_holds_only_the_folds_training_units(root):
    data, counts = patchtst.corpus(1, window_root=root, log=quiet)
    xjtu = windows.WindowStore(root / "xjtu_sy")
    split = protocols.xjtu_fold(xjtu.manifest, 1)
    assert counts["xjtu_sy"]["windows"] == int(split.train.sum()) == 9 * 20
    assert counts["cwru"]["windows"] == 3 * 4 * 2  # 0-2 hp only: 3 hp validates Gate 1
    assert "paderborn" not in counts  # no store here: left out, not an error
    assert data.shape == (sum(c["segments"] for c in counts.values()), 512)


def test_corpus_samples_windows_before_reading(root):
    data, counts = patchtst.corpus(2, window_root=root, max_segments=300, log=quiet)
    assert data.shape == (300, 512)
    assert counts["xjtu_sy"]["windows"] < 9 * 20  # only the sampled windows were read


def test_pretraining_resumes_and_refuses_other_settings(tmp_path, root):
    out = tmp_path / "fold1"
    kwargs = dict(
        fold=1, epochs=2, batch_size=64, max_segments=400, window_root=root, model_overrides=TINY
    )

    def stop_after_first_epoch(line):
        if line.startswith("[patchtst] epoch 1/"):
            raise KeyboardInterrupt  # a job killed at its time limit

    with pytest.raises(KeyboardInterrupt):
        patchtst.pretrain(out, log=stop_after_first_epoch, **kwargs)
    lines = []
    metrics = patchtst.pretrain(out, log=lines.append, **kwargs)
    assert any("resuming after epoch 1" in line for line in lines)
    assert [h["epoch"] for h in metrics["history"]] == [1, 2]
    assert np.isfinite(metrics["best_held_out"])
    encoder = get_encoder("patchtst", checkpoint=out / "model")
    assert encoder.encode(np.zeros((2, 512), np.float32) + np.arange(512)).shape == (2, 16)
    with pytest.raises(ValueError, match="other settings"):
        patchtst.pretrain(out, log=quiet, **{**kwargs, "batch_size": 32})


def test_fold_number_is_checked():
    with pytest.raises(SystemExit):
        patchtst.main(["--fold", "7"])


def test_a_dataset_with_nothing_allowed_is_left_out(root, tmp_path):
    from test_windows import paderborn_reader  # K001 (source), KA01 (source), KB27 (target)

    windows.build("paderborn", out=root / "paderborn", reader=paderborn_reader(), log=quiet)
    try:
        # Fold 1 tests K001 and KA01 and validates on fold 2's bearings: nothing is left.
        data, counts = patchtst.corpus(1, window_root=root, max_segments=300, log=quiet)
        assert counts["paderborn"] == {"windows": 0, "segments": 0}
        assert data.shape == (300, 512)  # the other datasets still fill the corpus
    finally:
        import shutil

        shutil.rmtree(root / "paderborn", ignore_errors=True)
