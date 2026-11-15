import hashlib
import json
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pdmfm import fetch
from pdmfm.sources import DatasetSource, RemoteFile

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_GET(self):
        if self.path == "/page.html.bin":
            body = b"<html>log in first</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start = 0
        header = self.headers.get("Range")
        if header:
            start = int(header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_download_verifies_size_and_hash(server, tmp_path):
    remote = RemoteFile(f"{server}/a.bin", "a.bin", size=len(PAYLOAD), sha256=DIGEST)
    assert fetch.obtain_file(remote, tmp_path / "a.bin")[0] == DIGEST
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD
    assert not (tmp_path / "a.bin.part").exists()


def test_download_resumes_a_partial_file(server, tmp_path):
    (tmp_path / "a.bin.part").write_bytes(PAYLOAD[:300_000])
    remote = RemoteFile(f"{server}/a.bin", "a.bin", size=len(PAYLOAD), sha256=DIGEST)
    fetch.obtain_file(remote, tmp_path / "a.bin")
    assert (tmp_path / "a.bin").read_bytes() == PAYLOAD


def test_wrong_hash_is_rejected_and_nothing_is_kept(server, tmp_path):
    remote = RemoteFile(f"{server}/a.bin", "a.bin", sha256="0" * 64)
    with pytest.raises(fetch.DownloadError, match="SHA-256"):
        fetch.obtain_file(remote, tmp_path / "a.bin")
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()


def test_html_instead_of_data_is_an_error(server, tmp_path):
    remote = RemoteFile(f"{server}/page.html.bin", "page.bin")
    with pytest.raises(fetch.DownloadError, match="HTML"):
        fetch.obtain_file(remote, tmp_path / "page.bin")


def test_existing_good_file_is_not_downloaded_again(tmp_path):
    (tmp_path / "a.bin").write_bytes(PAYLOAD)
    remote = RemoteFile("http://127.0.0.1:9/unreachable", "a.bin", sha256=DIGEST)
    assert fetch.obtain_file(remote, tmp_path / "a.bin")[0] == DIGEST


def test_fetch_dataset_writes_a_manifest(server, tmp_path, monkeypatch):
    source = DatasetSource(
        name="toy",
        title="toy",
        kind="direct",
        role="optional",
        licence="CC0",
        citation="none",
        homepage="https://example.org",
        files=(RemoteFile(f"{server}/a.bin", "sub/a.bin", sha256=DIGEST),),
    )
    monkeypatch.setitem(fetch.SOURCES, "toy", source)
    monkeypatch.setattr(fetch, "get_source", lambda name: fetch.SOURCES[name])
    target = fetch.fetch_dataset("toy", root=tmp_path)
    manifest = json.loads((target / "MANIFEST.json").read_text())
    assert manifest["licence"] == "CC0"
    assert manifest["files"]["sub/a.bin"]["sha256"] == DIGEST
    assert manifest["files"]["sub/a.bin"]["size"] == len(PAYLOAD)


def test_matching_local_seed_is_copied_without_network(tmp_path):
    seed = tmp_path / "seed" / "a.bin"
    seed.parent.mkdir()
    seed.write_bytes(PAYLOAD)
    remote = RemoteFile("http://127.0.0.1:9/unreachable", "a.bin", sha256=DIGEST)
    digest, origin = fetch.obtain_file(remote, tmp_path / "out" / "a.bin", seed=seed)
    assert (digest, origin) == (DIGEST, "seed")
    assert (tmp_path / "out" / "a.bin").read_bytes() == PAYLOAD


def test_mismatched_seed_falls_back_to_download(server, tmp_path):
    seed = tmp_path / "a.bin.seed"
    seed.write_bytes(PAYLOAD.replace(b"\n", b"\r\n"))  # e.g. line endings changed by Windows
    remote = RemoteFile(f"{server}/a.bin", "a.bin", sha256=DIGEST)
    digest, origin = fetch.obtain_file(remote, tmp_path / "a.bin", seed=seed)
    assert (digest, origin) == (DIGEST, "download")


def test_seed_is_ignored_without_a_publisher_hash(server, tmp_path):
    seed = tmp_path / "seed.bin"
    seed.write_bytes(b"something else")
    remote = RemoteFile(f"{server}/a.bin", "a.bin")
    assert fetch.obtain_file(remote, tmp_path / "a.bin", seed=seed)[1] == "download"


def test_exclude_skips_matching_paths(server, tmp_path, monkeypatch):
    files = (
        RemoteFile(f"{server}/t.bin", "tool/t.bin", sha256=DIGEST),
        RemoteFile(f"{server}/s.bin", "spec/x/s.bin", sha256=DIGEST),
    )
    source = DatasetSource(
        name="toy",
        title="toy",
        kind="direct",
        role="optional",
        licence="CC0",
        citation="none",
        homepage="https://example.org",
        files=files,
    )
    monkeypatch.setitem(fetch.SOURCES, "toy", source)
    monkeypatch.setattr(fetch, "get_source", lambda name: fetch.SOURCES[name])
    fetch.fetch_dataset("toy", root=tmp_path, exclude=("spec/",))
    assert (tmp_path / "tool" / "t.bin").exists()
    assert not (tmp_path / "spec").exists()


def test_parallel_fetch_records_every_file(server, tmp_path, monkeypatch):
    files = tuple(
        RemoteFile(f"{server}/f{i}.bin", f"dir/f{i}.bin", sha256=DIGEST) for i in range(5)
    )
    source = DatasetSource(
        name="toy",
        title="toy",
        kind="direct",
        role="optional",
        licence="CC0",
        citation="none",
        homepage="https://example.org",
        files=files,
    )
    monkeypatch.setitem(fetch.SOURCES, "toy", source)
    monkeypatch.setattr(fetch, "get_source", lambda name: fetch.SOURCES[name])
    target = fetch.fetch_dataset("toy", root=tmp_path, workers=4)
    manifest = json.loads((target / "MANIFEST.json").read_text())
    assert sorted(manifest["files"]) == sorted(f.relpath for f in files)
    assert all((tmp_path / f.relpath).read_bytes() == PAYLOAD for f in files)


def test_mendeley_listing_builds_folder_paths():
    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    folders = [
        {"id": "f1", "name": "spec"},
        {"id": "f2", "name": "x", "parent_id": "f1"},
    ]

    def entry(name):
        return {
            "filename": name,
            "size": 3,
            "content_details": {"download_url": f"https://m/{name}", "sha256_hash": "ab"},
        }

    files = {"root": [entry("labels.csv")], "f1": [], "f2": [entry("T1R1B1.jpg")]}

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            if url.endswith("/folders/1"):
                return FakeResponse(folders)
            return FakeResponse(files[params["folder_id"]])

    listed = fetch._mendeley_files(FakeSession(), "abc", 1)
    assert [f.relpath for f in listed] == ["labels.csv", "spec/x/T1R1B1.jpg"]
    assert listed[1].sha256 == "ab" and listed[1].size == 3


def test_extract_unzips_and_copies(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "vibration.zip", "w") as archive:
        archive.writestr("vibration/0Nm_Normal.mat", b"data")
    (raw / "readme.pdf").write_bytes(b"%PDF")
    (raw / "MANIFEST.json").write_text("{}")
    out = fetch.extract_dataset("toy", root=raw, out=tmp_path / "out")
    assert (out / "vibration" / "0Nm_Normal.mat").read_bytes() == b"data"
    assert (out / "readme.pdf").exists()
    assert not (out / "MANIFEST.json").exists()


def test_zips_with_clashing_member_names_get_their_own_folders(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, payload in (("vibration.zip", b"vib"), ("acoustic.zip", b"mic")):
        with zipfile.ZipFile(raw / name, "w") as archive:
            archive.writestr("0Nm_BPFI_03.mat", payload)  # same member name in both
    out = fetch.extract_dataset("toy", root=raw, out=tmp_path / "out")
    assert (out / "vibration" / "0Nm_BPFI_03.mat").read_bytes() == b"vib"
    assert (out / "acoustic" / "0Nm_BPFI_03.mat").read_bytes() == b"mic"


def _manual_source(monkeypatch):
    source = DatasetSource(
        name="toy",
        title="toy",
        kind="manual",
        role="target",
        licence="none stated",
        citation="none",
        homepage="https://example.org",
        params={"required": ["c1.zip", "c4.zip"]},
    )
    monkeypatch.setitem(fetch.SOURCES, "toy", source)
    monkeypatch.setattr(fetch, "get_source", lambda name: fetch.SOURCES[name])


def test_verify_reports_missing_required_files(tmp_path, monkeypatch):
    _manual_source(monkeypatch)
    (tmp_path / "c1.zip").write_bytes(b"one")
    report = fetch.verify_dataset("toy", root=tmp_path)
    assert not report["ok"] and report["missing"] == ["c4.zip"]
    assert report["added"] == ["c1.zip"]


def test_verify_records_hand_placed_files_then_detects_changes(tmp_path, monkeypatch):
    _manual_source(monkeypatch)
    for name in ("c1.zip", "c4.zip"):
        (tmp_path / name).write_bytes(name.encode())
    first = fetch.verify_dataset("toy", root=tmp_path)
    assert first["ok"] and sorted(first["added"]) == ["c1.zip", "c4.zip"]
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["files"]["c1.zip"]["origin"] == "manual"
    second = fetch.verify_dataset("toy", root=tmp_path)
    assert second["ok"] and second["checked"] == 2 and second["added"] == []
    (tmp_path / "c4.zip").write_bytes(b"altered after copying")
    third = fetch.verify_dataset("toy", root=tmp_path)
    assert not third["ok"] and third["mismatch"] == ["c4.zip"]


def test_verify_cli_exits_non_zero_when_incomplete(tmp_path, monkeypatch):
    _manual_source(monkeypatch)
    monkeypatch.setattr(fetch, "raw_dir", lambda name: tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        fetch.main(["verify", "phm2010"])
    assert exit_info.value.code == 1


def test_rar_tool_choice_and_destination_separator(monkeypatch, tmp_path):
    archive, destination = tmp_path / "K001.rar", tmp_path / "out"
    tools = {}
    monkeypatch.setattr(fetch.shutil, "which", lambda name: tools.get(name))
    monkeypatch.setattr(fetch, "WINRAR_UNRAR", ())
    tools["7z"] = "/usr/bin/7z"
    assert fetch._rar_command(archive, destination)[:3] == ["7z", "x", "-y"]
    tools.clear()
    tools["unrar"] = "/usr/bin/unrar"
    command = fetch._rar_command(archive, destination)
    assert command[0] == "/usr/bin/unrar" and command[-1] == str(destination) + os.sep
    tools.clear()
    assert "-xf" in fetch._rar_command(archive, destination)  # the system tar, last resort
