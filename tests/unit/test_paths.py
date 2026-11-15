from pdmfm import paths


def test_data_root_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.raw_dir("kaist") == tmp_path / "raw" / "kaist"
    assert paths.extracted_dir("kaist") == tmp_path / "extracted" / "kaist"


def test_default_data_root_is_inside_the_repository(monkeypatch):
    monkeypatch.delenv("PDMFM_DATA_ROOT", raising=False)
    assert paths.data_root() == paths.REPO_ROOT / "data"
