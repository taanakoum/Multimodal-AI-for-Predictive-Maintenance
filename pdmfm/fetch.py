"""Download and unpack the Project 14 datasets.

Usage::

    python -m pdmfm.fetch list
    python -m pdmfm.fetch plan kaist            # files, sizes, hashes; downloads nothing
    python -m pdmfm.fetch get kaist             # resumable download into raw/kaist
    python -m pdmfm.fetch get mbhm --include-large
    python -m pdmfm.fetch extract kaist         # zip and rar archives into extracted/kaist
    python -m pdmfm.fetch verify phm2010        # hand-placed or copied files: hash and check

Every completed file is checked against the publisher's size and SHA-256 when they are
known, and recorded in ``raw/<dataset>/MANIFEST.json`` with its own SHA-256, so a copy on
another machine (Ruche) can be verified file by file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

from pdmfm.paths import extracted_dir, raw_dir
from pdmfm.sources import SOURCES, DatasetSource, RemoteFile, get_source

USER_AGENT = "Mozilla/5.0 (pdmfm data fetcher; research use)"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, */*"})
    return session


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- listing


def _mendeley_files(session: requests.Session, dataset_id: str, version: int) -> list[RemoteFile]:
    folders = session.get(f"{MENDELEY_API}/{dataset_id}/folders/{version}", timeout=60)
    folders.raise_for_status()
    by_id = {folder["id"]: folder for folder in folders.json()}

    def folder_path(folder_id: str) -> str:
        parts = []
        while folder_id in by_id:
            parts.append(by_id[folder_id]["name"])
            folder_id = by_id[folder_id].get("parent_id", "")
        return "/".join(reversed(parts))

    remote: list[RemoteFile] = []
    for folder_id in ["root", *by_id]:
        response = session.get(
            f"{MENDELEY_API}/{dataset_id}/files",
            params={"folder_id": folder_id, "version": version},
            timeout=60,
        )
        response.raise_for_status()
        prefix = "" if folder_id == "root" else folder_path(folder_id) + "/"
        for entry in response.json():
            details = entry["content_details"]
            remote.append(
                RemoteFile(
                    url=details["download_url"],
                    relpath=prefix + entry["filename"],
                    size=entry["size"],
                    sha256=details.get("sha256_hash"),
                )
            )
    return remote


def _huggingface_files(
    session: requests.Session, repo: str, wanted: Iterable[str]
) -> list[RemoteFile]:
    response = session.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", timeout=60)
    response.raise_for_status()
    entries = {entry["path"]: entry for entry in response.json() if entry["type"] == "file"}
    remote = []
    for path in wanted:
        if path not in entries:
            raise DownloadError(f"{repo} has no file {path!r}")
        entry = entries[path]
        lfs = entry.get("lfs") or {}
        remote.append(
            RemoteFile(
                url=f"https://huggingface.co/datasets/{repo}/resolve/main/{path}",
                relpath=path,
                size=entry["size"],
                sha256=lfs.get("oid"),  # git-LFS object ids are SHA-256 of the content
            )
        )
    return remote


def list_remote_files(
    source: DatasetSource, session: requests.Session | None = None, include_large: bool = False
) -> list[RemoteFile]:
    """Files to download for ``source``, with publisher sizes and hashes when known."""
    session = session or make_session()
    if source.kind == "mendeley":
        return _mendeley_files(session, source.params["dataset_id"], source.params["version"])
    if source.kind == "huggingface":
        wanted = list(source.params["files"])
        if include_large:
            wanted += list(source.params.get("large_files", []))
        return _huggingface_files(session, source.params["repo"], wanted)
    if source.kind in ("direct", "gdrive"):
        return list(source.files)
    return []  # manual: placed by hand, see source.notes


# ----------------------------------------------------------------------------- downloading


def _verify(path: Path, remote: RemoteFile) -> str:
    """Check size and hash against the publisher's values; return the file's SHA-256."""
    if remote.size is not None and path.stat().st_size != remote.size:
        raise DownloadError(
            f"{remote.relpath}: size {path.stat().st_size} != expected {remote.size}"
        )
    digest = sha256_of(path)
    if remote.sha256 is not None and digest != remote.sha256.lower():
        raise DownloadError(f"{remote.relpath}: SHA-256 {digest} != expected {remote.sha256}")
    return digest


def _download_http(session: requests.Session, remote: RemoteFile, part: Path) -> None:
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(remote.url, headers=headers, stream=True, timeout=120) as response:
        if response.status_code == 416:  # already complete
            return
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type and not remote.relpath.endswith((".html", ".htm")):
            raise DownloadError(
                f"{remote.relpath}: server returned an HTML page instead of the file "
                f"({remote.url}); the link may need a login or may be dead"
            )
        resumed = response.status_code == 206
        with part.open("ab" if resumed else "wb") as handle:
            for block in response.iter_content(CHUNK):
                handle.write(block)


def _download_gdrive(remote: RemoteFile, part: Path) -> None:
    import gdown  # imported lazily: only Google Drive sources (XJTU-SY) need it

    file_id = remote.url.removeprefix("gdrive:")
    result = gdown.download(id=file_id, output=str(part), quiet=False, resume=True)
    if result is None:
        raise DownloadError(f"{remote.relpath}: Google Drive refused the download (quota?)")


def obtain_file(
    remote: RemoteFile,
    destination: Path,
    session: requests.Session | None = None,
    seed: Path | None = None,
) -> tuple[str, str]:
    """Make ``destination`` hold ``remote``; return (SHA-256, origin).

    The origin is ``"existing"`` (already there and valid), ``"seed"`` (copied from a local
    copy) or ``"download"``. A ``seed`` is used instead of the network only when the
    publisher's SHA-256 is known and the seed matches it exactly.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            return _verify(destination, remote), "existing"
        except DownloadError:
            destination.unlink()
    if seed is not None and seed.is_file() and remote.sha256 is not None:
        try:
            digest = _verify(seed, remote)
        except DownloadError:
            pass  # the local copy differs from the published file: download instead
        else:
            shutil.copy2(seed, destination)
            return digest, "seed"
    part = destination.with_name(destination.name + ".part")
    if remote.url.startswith("gdrive:"):
        _download_gdrive(remote, part)
    else:
        _download_http(session or make_session(), remote, part)
    try:
        digest = _verify(part, remote)
    except DownloadError:
        part.unlink(missing_ok=True)
        raise
    part.replace(destination)
    return digest, "download"


def download_file(
    remote: RemoteFile,
    destination: Path,
    session: requests.Session | None = None,
    seed: Path | None = None,
) -> str:
    """Download one file to ``destination`` (resumable); return its SHA-256."""
    return obtain_file(remote, destination, session, seed)[0]


def _read_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def fetch_dataset(
    name: str,
    include_large: bool = False,
    only: str | None = None,
    session: requests.Session | None = None,
    root: Path | None = None,
    workers: int = 1,
    seed_from: Path | None = None,
    exclude: tuple[str, ...] = (),
) -> Path:
    """Download every file of dataset ``name`` into its raw folder and update the manifest.

    ``workers`` > 1 downloads HTTP files in parallel (useful for Nonastreda's 4,611 small
    files); Google Drive files are always fetched one at a time to stay within its quota.
    ``seed_from`` is a local folder with the same layout whose files are copied instead of
    downloaded when they match the publisher's SHA-256. Paths containing any ``exclude``
    text are skipped.
    """
    source = get_source(name)
    target = root or raw_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    if source.kind == "manual":
        raise DownloadError(f"{name} must be placed by hand in {target}: {source.notes}")
    session = session or make_session()
    remote_files = list_remote_files(source, session, include_large=include_large)
    if only:
        remote_files = [f for f in remote_files if only in f.relpath]
    if exclude:
        remote_files = [f for f in remote_files if not any(x in f.relpath for x in exclude)]
    manifest_path = target / "MANIFEST.json"
    manifest = _read_manifest(manifest_path)
    manifest.update(
        dataset=name, licence=source.licence, citation=source.citation, homepage=source.homepage
    )
    entries = manifest.setdefault("files", {})
    parallel = workers > 1 and source.kind != "gdrive"
    lock = threading.Lock()
    local = threading.local()

    def worker_session() -> requests.Session:
        if not parallel:
            return session
        if not hasattr(local, "session"):  # requests sessions are not shared across threads
            local.session = make_session()
        return local.session

    def fetch_one(item: tuple[int, RemoteFile]) -> None:
        index, remote = item
        size = f"{remote.size / 1e6:.1f} MB" if remote.size else "size unknown"
        print(f"[{name} {index}/{len(remote_files)}] {remote.relpath} ({size})", flush=True)
        seed = seed_from / remote.relpath if seed_from is not None else None
        digest, origin = obtain_file(remote, target / remote.relpath, worker_session(), seed)
        previous = entries.get(remote.relpath, {})
        entry = {
            "url": remote.url,
            "origin": previous.get("origin", origin) if origin == "existing" else origin,
            "copied_from": str(seed) if origin == "seed" else previous.get("copied_from"),
            "size": (target / remote.relpath).stat().st_size,
            "sha256": digest,
            "publisher_sha256": remote.sha256,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with lock:
            entries[remote.relpath] = entry
            if len(entries) % 25 == 0:
                _write_manifest(manifest_path, manifest)

    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fetch_one, enumerate(remote_files, 1)))
        else:
            for item in enumerate(remote_files, 1):
                fetch_one(item)
    finally:
        _write_manifest(manifest_path, manifest)  # keep what finished, even after an error
    return target


# ----------------------------------------------------------------------------- extraction


WINRAR_UNRAR = (
    Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
    Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe"),
)


def _rar_command(archive: Path, destination: Path) -> list[str]:
    """Pick a RAR extractor: 7-Zip, then UnRAR (WinRAR), then the system ``tar``.

    Only 7-Zip and UnRAR read multi-volume RAR5 (XJTU-SY). Windows ``tar`` (libarchive)
    handles single-volume archives and stops with an error at the end of a first volume.
    """
    for tool in ("7z", "7za"):
        if shutil.which(tool):
            return [tool, "x", "-y", f"-o{destination}", str(archive)]
    unrar = shutil.which("unrar") or next((str(p) for p in WINRAR_UNRAR if p.exists()), None)
    if unrar:  # a trailing separator tells unrar the destination is a folder
        return [unrar, "x", "-o+", "-y", "-idq", str(archive), str(destination) + os.sep]
    tar = Path(r"C:\Windows\System32\tar.exe")
    return [str(tar) if tar.exists() else "tar", "-xf", str(archive), "-C", str(destination)]


def _extract_rar(archive: Path, destination: Path) -> None:
    command = _rar_command(archive, destination)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(f"could not extract {archive.name}: {result.stderr.strip()[:500]}")


def _zip_target(archive: zipfile.ZipFile, parent: Path, path: Path) -> Path:
    """Where a zip unpacks: into ``parent`` when all its members share one top folder,
    otherwise into ``parent/<zip stem>`` so that two archives with the same member names
    (KAIST's vibration.zip and acoustic.zip) cannot overwrite each other."""
    tops = {Path(name).parts[0] for name in archive.namelist() if Path(name).parts}
    single_folder = len(tops) == 1 and all(
        len(Path(name).parts) > 1 or name.endswith("/") for name in archive.namelist()
    )
    return parent if single_folder else parent / path.stem


def extract_dataset(name: str, root: Path | None = None, out: Path | None = None) -> Path:
    """Unpack every zip and rar archive of ``name``; non-archive files are copied as they are."""
    source_dir = root or raw_dir(name)
    destination = out or extracted_dir(name)
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.name in ("MANIFEST.json",) or path.suffix == ".part":
            continue
        relative = path.relative_to(source_dir)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(_zip_target(archive, destination / relative.parent, path))
        elif path.suffix.lower() == ".rar":
            if ".part" in path.stem and not path.stem.endswith(("part01", "part1")):
                continue  # later volumes are read through the first one
            target = destination / relative.parent
            target.mkdir(parents=True, exist_ok=True)
            _extract_rar(path, target)
        else:
            copy = destination / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            if not copy.exists():
                shutil.copy2(path, copy)
    return destination


# ----------------------------------------------------------------------------- verification


def verify_dataset(name: str, root: Path | None = None) -> dict:
    """Check a dataset's raw folder against its manifest (and its required files).

    * Files in the manifest are re-hashed: a different SHA-256 is a ``mismatch``.
    * Files on disk but not in the manifest (placed by hand, e.g. PHM 2010 zips or a
      Paderborn copy brought back from Ruche) are hashed and added with origin ``manual``.
    * For manual sources, ``params["required"]`` files must be present.

    Returns a report dict with ``ok`` False when anything is missing or mismatched.
    """
    source = get_source(name)
    target = root or raw_dir(name)
    manifest_path = target / "MANIFEST.json"
    manifest = _read_manifest(manifest_path)
    manifest.update(
        dataset=name, licence=source.licence, citation=source.citation, homepage=source.homepage
    )
    entries = manifest.setdefault("files", {})
    report = {"dataset": name, "checked": 0, "added": [], "mismatch": [], "missing": []}
    on_disk = (
        {
            p.relative_to(target).as_posix(): p
            for p in target.rglob("*")
            if p.is_file() and p.name != "MANIFEST.json" and not p.name.endswith((".part", ".tmp"))
        }
        if target.is_dir()
        else {}
    )
    for relpath, entry in sorted(entries.items()):
        path = on_disk.get(relpath)
        if path is None:
            report["missing"].append(relpath)
            continue
        report["checked"] += 1
        if sha256_of(path) != entry["sha256"]:
            report["mismatch"].append(relpath)
    for relpath, path in sorted(on_disk.items()):
        if relpath in entries:
            continue
        entries[relpath] = {
            "url": None,
            "origin": "manual",
            "size": path.stat().st_size,
            "sha256": sha256_of(path),
            "publisher_sha256": None,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        report["added"].append(relpath)
    for required in source.params.get("required", []):
        if required not in on_disk:
            report["missing"].append(required)
    if target.is_dir():
        _write_manifest(manifest_path, manifest)
    report["ok"] = not report["missing"] and not report["mismatch"]
    return report


# ----------------------------------------------------------------------------- command line


def _cmd_verify(args: argparse.Namespace) -> None:
    report = verify_dataset(args.dataset)
    print(
        f"{args.dataset}: {report['checked']} files verified, {len(report['added'])} added, "
        f"{len(report['mismatch'])} mismatched, {len(report['missing'])} missing"
    )
    for key in ("mismatch", "missing", "added"):
        for relpath in report[key]:
            print(f"  {key}: {relpath}")
    if not report["ok"]:
        raise DownloadError(f"{args.dataset} is incomplete or altered")


def _cmd_list(_: argparse.Namespace) -> None:
    for source in SOURCES.values():
        print(f"{source.name:15s} {source.role:13s} {source.kind:11s} {source.licence}")


def _cmd_plan(args: argparse.Namespace) -> None:
    source = get_source(args.dataset)
    files = list_remote_files(source, include_large=args.include_large)
    total = sum(f.size or 0 for f in files)
    for f in files:
        size = f"{f.size / 1e6:10.1f} MB" if f.size else "   unknown"
        print(f"{size}  {f.relpath}")
    print(f"{len(files)} files, {total / 1e9:.2f} GB known; licence: {source.licence}")
    if source.notes:
        print(f"note: {source.notes}")


def _cmd_get(args: argparse.Namespace) -> None:
    path = fetch_dataset(
        args.dataset,
        include_large=args.include_large,
        only=args.only,
        workers=args.workers,
        seed_from=args.seed_from,
        exclude=tuple(args.exclude),
    )
    print(f"done: {path}")


def _cmd_extract(args: argparse.Namespace) -> None:
    print(f"done: {extract_dataset(args.dataset)}")


def main(argv: list[str] | None = None) -> None:
    summary = (__doc__ or "pdmfm-fetch").splitlines()[0]  # docstrings are gone under -OO
    parser = argparse.ArgumentParser(prog="pdmfm-fetch", description=summary)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the registered datasets").set_defaults(func=_cmd_list)
    for command, func in (("plan", _cmd_plan), ("get", _cmd_get)):
        p = sub.add_parser(command)
        p.add_argument("dataset", choices=sorted(SOURCES))
        p.add_argument("--include-large", action="store_true")
        if command == "get":
            p.add_argument("--only", help="download only files whose path contains this text")
            p.add_argument("--workers", type=int, default=1, help="parallel HTTP downloads")
            p.add_argument("--seed-from", type=Path, help="local copy to reuse when hashes match")
            p.add_argument(
                "--exclude", action="append", default=[], help="skip paths containing this"
            )
        p.set_defaults(func=func)
    p = sub.add_parser("extract")
    p.add_argument("dataset", choices=sorted(SOURCES))
    p.set_defaults(func=_cmd_extract)
    p = sub.add_parser("verify", help="re-hash files against MANIFEST.json; add hand-placed files")
    p.add_argument("dataset", choices=sorted(SOURCES))
    p.set_defaults(func=_cmd_verify)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
