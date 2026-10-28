"""The feature cache (pdmfm.features) with the handcrafted encoder and stand-ins."""

from __future__ import annotations

import time

import numpy as np
import pytest
from synthetic import cwru_reader

from pdmfm import arms, features, protocols, texts, windows
from pdmfm.encoders import StatsEncoder


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    return windows.build("cwru", reader=cwru_reader(), log=lambda *_: None)


@pytest.fixture
def cache(tmp_path):
    return features.FeatureStore(tmp_path / "features")


def quiet(*_):
    pass


def online(store, encoder, arm, rows):
    x = arms.arm_input(store, rows, arm)
    n, c, k, length = x.shape
    return arms.pool_tokens(encoder.encode(x.reshape(-1, length)).reshape(n, c, k, -1), arm)


@pytest.mark.parametrize("arm", ["T", "F", "E"])
def test_cached_features_equal_online_encoding(store, cache, arm):
    encoder = StatsEncoder()
    cached = cache.extract(store, encoder, arm, chunk=7, log=quiet)
    assert cached.done.all() and cached.meta["complete"]
    tokens = 6 if arm == "T" else 1
    channels = 2  # both CWRU channels are vibration, so E uses both
    assert cached.features.shape == (len(store), channels, tokens, 36)
    rows = np.array([0, 9, 31])
    expected = online(store, encoder, arm, rows)
    np.testing.assert_allclose(
        np.asarray(cached.features[rows], dtype=np.float32), expected, rtol=2e-3, atol=2e-3
    )
    pooled = cached.pooled(rows)
    assert pooled.shape == (3, channels * 36)
    assert cache.find(store, "stats", arm).folder == cached.folder


class Failing(StatsEncoder):
    """Stats encoder that fails after a number of calls (a job killed mid-way)."""

    def __init__(self, calls: int) -> None:
        self.calls = calls

    def encode(self, x):
        self.calls -= 1
        if self.calls < 0:
            raise RuntimeError("killed")
        return super().encode(x)


def test_extraction_resumes_where_it_stopped(store, cache, tmp_path):
    with pytest.raises(RuntimeError, match="killed"):
        cache.extract(store, Failing(calls=2), "F", chunk=5, log=quiet)
    partial = cache.find(store, "stats", "F")
    assert partial.done.sum() == 10 and not partial.meta["complete"]  # 2 chunks saved
    calls = []

    class Counting(StatsEncoder):
        def encode(self, x):
            calls.append(len(x))
            return super().encode(x)

    resumed = cache.extract(store, Counting(), "F", chunk=5, log=quiet)
    assert resumed.done.all() and sum(calls) == (len(store) - 10) * 2  # only the rest
    clean = features.FeatureStore(tmp_path / "clean").extract(store, StatsEncoder(), "F", log=quiet)
    np.testing.assert_array_equal(np.asarray(resumed.features), np.asarray(clean.features))


def test_subsample_then_complete(store, cache):
    rows = protocols.subsample(store.manifest, 0.25, seed=0)
    part = cache.extract(store, StatsEncoder(), "F", rows=rows, log=quiet)
    assert (part.done == rows).all() and not part.meta["complete"]
    full = cache.extract(store, StatsEncoder(), "F", log=quiet)
    assert full.done.all() and full.folder == part.folder


def test_other_encoder_signature_gets_another_folder(store, cache):
    class OtherCheckpoint(StatsEncoder):
        def signature(self):
            return {**super().signature(), "variant": "other checkpoint"}

    first = cache.extract(store, StatsEncoder(), "F", log=quiet)
    time.sleep(1.1)  # "updated" has one-second resolution
    second = cache.extract(store, OtherCheckpoint(), "F", log=quiet)
    assert first.folder != second.folder
    with pytest.warns(UserWarning, match="several"):
        assert cache.find(store, "stats", "F").folder == second.folder  # the most recent
    with pytest.raises(FileNotFoundError):
        cache.find(store, "mantis", "F")


def test_a_complete_cache_wins_over_a_newer_incomplete_one(store, cache):
    class Retrained(StatsEncoder):
        def signature(self):
            return {**super().signature(), "variant": "retrained"}

    complete = cache.extract(store, StatsEncoder(), "E", log=quiet)
    time.sleep(1.1)
    cache.extract(store, Retrained(), "E", rows=np.arange(3), log=quiet)  # interrupted early
    with pytest.warns(UserWarning):
        assert cache.find(store, "stats", "E").folder == complete.folder


def test_waveform_encoders_take_the_s_arm_only(store, cache):
    class Waveform(StatsEncoder):
        input_kind = "waveform"

    with pytest.raises(ValueError, match="cannot take arm"):
        cache.extract(store, Waveform(), "F", log=quiet)
    with pytest.raises(ValueError, match="cannot take arm"):
        cache.extract(store, StatsEncoder(), "S", log=quiet)


def test_status_lists_progress(store, cache):
    cache.extract(store, StatsEncoder(), "E", rows=np.arange(4), log=quiet)
    table = cache.status()
    assert table.loc[0, ["dataset", "arm", "encoder", "done", "windows"]].tolist() == [
        "cwru",
        "E",
        "stats",
        4,
        32,
    ]
    assert not table.loc[0, "complete"]


class FakeImageEncoder:
    name = "fakeimg"

    def signature(self, kind):
        return {"encoder": self.name, "kind": kind}

    def encode_paths(self, paths, kind):
        return np.ones((len(paths), 5, 8), dtype=np.float32)


def test_image_cache(tmp_path, monkeypatch, cache):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(
        features, "image_paths", lambda d: {"tool": ["raw/a.png", "raw/b.png"], "chip": []}
    )
    first = features.extract_images("nonastreda", FakeImageEncoder(), root=cache.root, log=quiet)
    assert len(first) == 1 and np.load(first[0] / "features.npy").shape == (2, 5, 8)
    assert (first[0] / "paths.csv").read_text().splitlines() == ["path", "raw/a.png", "raw/b.png"]
    assert (
        features.extract_images("nonastreda", FakeImageEncoder(), root=cache.root, log=quiet)
        == first
    )


class FakeTextEncoder:
    name = "faketext"
    dim = 6

    def signature(self):
        return {"encoder": self.name}

    def encode(self, sentences):
        rng = [
            np.random.default_rng(abs(hash(s)) % 2**32).standard_normal(self.dim) for s in sentences
        ]
        z = np.array(rng, dtype=np.float32)
        return z / np.linalg.norm(z, axis=1, keepdims=True)


def test_text_cache_prototypes_and_contexts(store, cache):
    (path,) = features.extract_texts(FakeTextEncoder(), ["cwru"], root=cache.root, log=quiet)
    data = np.load(path)
    assert list(data["classes"]) == list(texts.load("cwru").descriptions)
    np.testing.assert_allclose(np.linalg.norm(data["prototypes"], axis=1), 1.0, rtol=1e-5)
    assert len(data["contexts"]) == 4 and data["context_emb"].shape == (4, 6)  # one per load
    assert features.extract_texts(FakeTextEncoder(), ["cwru"], root=cache.root, log=quiet) == [path]


def test_encoder_input_rate_must_match_the_store(store, cache):
    class At16k(StatsEncoder):
        input_kind = "waveform"
        input_rate = 16_000.0

    with pytest.raises(ValueError, match="expects 16000"):
        cache.extract(store, At16k(), "S", log=quiet)
