"""Registry of the datasets used by Project 14: where they come from and under which terms.

Each entry says how to obtain the files (``kind``), the licence, the citation the
publishers ask for, and the role the dataset plays in the plan (pretraining source,
held-out target, or optional). Facts were checked on the publishers' pages on
2026-10-25; see ``PROJECT_LOG.md`` for the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["mendeley", "direct", "gdrive", "huggingface", "manual"]
Role = Literal["source", "target", "source+target", "optional"]


@dataclass(frozen=True)
class RemoteFile:
    """One file to download; ``relpath`` is relative to the dataset's raw folder."""

    url: str
    relpath: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    name: str
    title: str
    kind: Kind
    role: Role
    licence: str
    citation: str
    homepage: str
    params: dict = field(default_factory=dict)
    files: tuple[RemoteFile, ...] = ()
    notes: str = ""


# CWRU 12 kHz drive-end fault files and the 48 kHz normal baseline: the classic
# 10-class set (normal; inner race, ball, outer race at 6 o'clock; 0.007/0.014/0.021 in).
# Each group is four consecutive file numbers, one per motor load 0-3 hp.
CWRU_GROUPS: tuple[tuple[str, float, int], ...] = (
    ("normal", 0.0, 97),
    ("inner_race", 0.007, 105),
    ("ball", 0.007, 118),
    ("outer_race", 0.007, 130),
    ("inner_race", 0.014, 169),
    ("ball", 0.014, 185),
    ("outer_race", 0.014, 197),
    ("inner_race", 0.021, 209),
    ("ball", 0.021, 222),
    ("outer_race", 0.021, 234),
)
# File number -> (class, fault size in inches, motor load in hp).
CWRU_FILES: dict[int, tuple[str, float, int]] = {
    first + load: (kind, size, load) for kind, size, first in CWRU_GROUPS for load in range(4)
}

# Paderborn bearing codes (Lessmeier et al., PHME 2016, Tables 4, 5 and 7).
PADERBORN_HEALTHY = ("K001", "K002", "K003", "K004", "K005", "K006")
PADERBORN_ARTIFICIAL = (
    "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KI01", "KI03", "KI05", "KI07", "KI08",
)
PADERBORN_REAL = (
    "KA04", "KA15", "KA16", "KA22", "KA30", "KB23", "KB24", "KB27",
    "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
)
# Plan split: healthy K001-K003 join the source pool, K004-K006 the real-damage target.
PADERBORN_SOURCE = PADERBORN_HEALTHY[:3] + PADERBORN_ARTIFICIAL
PADERBORN_TARGET = PADERBORN_HEALTHY[3:] + PADERBORN_REAL
PADERBORN_SETTINGS = {
    "N15_M07_F10": (1500, 0.7, 1000),
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}

# XJTU-SY multi-volume RAR parts in the authors' Google Drive folder.
XJTU_SY_PARTS = {
    "XJTU-SY_Bearing_Datasets.part01.rar": "1ATvZuD6j3bPxhyR07Zm-PURmOC4b4uRn",
    "XJTU-SY_Bearing_Datasets.part02.rar": "162KvWNIpBGtd7EDWo4yP1j5XsaoNHOYU",
    "XJTU-SY_Bearing_Datasets.part03.rar": "1NvzrGW-KOSy48OZmiFxlE3TPV4CKAcw0",
    "XJTU-SY_Bearing_Datasets.part04.rar": "1VuQ5-mK11p1S2pTxUZaH_IxOwUlsmN0S",
    "XJTU-SY_Bearing_Datasets.part05.rar": "1WH4OU4MLaMGQkbh6DghxPA5Dwvsq8tEf",
    "XJTU-SY_Bearing_Datasets.part06.rar": "1wzQzQUx6-J8DuGczT81OkrkTgOUwL-I_",
}

SOURCES: dict[str, DatasetSource] = {
    "nonastreda": DatasetSource(
        name="nonastreda",
        title="Nonastreda: multimodal dataset for identifying tool wear condition",
        kind="mendeley",
        role="source",
        licence="CC BY 4.0",
        citation="H. Truchan, Z. Ahmadi, Nonastreda multimodal dataset for efficient tool wear "
        "state monitoring, Data in Brief (2025). doi:10.17632/m892d2wtzh.1",
        homepage="https://data.mendeley.com/datasets/m892d2wtzh/1",
        params={"dataset_id": "m892d2wtzh", "version": 1},
        notes="Force 1 kHz (Kistler 9257B); tool, chip and workpiece photos per pass; 1.41 GB.",
    ),
    "kaist": DatasetSource(
        name="kaist",
        title="Vibration, acoustic, temperature and motor current dataset of a rotating machine "
        "under varying load conditions (KAIST)",
        kind="mendeley",
        role="target",
        licence="CC BY 4.0",
        citation="W. Jung et al., Data in Brief 48 (2023) 109049. doi:10.17632/ztmf3m7h5x.6",
        homepage="https://data.mendeley.com/datasets/ztmf3m7h5x/6",
        params={"dataset_id": "ztmf3m7h5x", "version": 6},
        notes="Load-varying set: vibration.zip, current,temp.zip, acoustic.zip (4.25 GB).",
    ),
    "najafi_thermal": DatasetSource(
        name="najafi_thermal",
        title="Thermal image of equipment (induction motor)",
        kind="mendeley",
        role="optional",
        licence="CC BY 4.0",
        citation="M. Najafi et al., ICSPIS 2020, doi:10.1109/ICSPIS51611.2020.9349599; "
        "dataset doi:10.17632/m4sbt8hbvk.3",
        homepage="https://data.mendeley.com/datasets/m4sbt8hbvk/3",
        params={"dataset_id": "m4sbt8hbvk", "version": 3},
        notes="Two RAR archives (BMP images, ground truths) and a readme PDF; 11 MB.",
    ),
    "cwru": DatasetSource(
        name="cwru",
        title="Case Western Reserve University bearing data",
        kind="direct",
        role="source",
        licence="Public download; no licence stated (cite the Bearing Data Center)",
        citation="Case Western Reserve University Bearing Data Center, "
        "https://engineering.case.edu/bearingdatacenter",
        homepage="https://engineering.case.edu/bearingdatacenter",
        files=tuple(
            RemoteFile(
                url=f"https://engineering.case.edu/sites/default/files/{number}.mat",
                relpath=f"{number}.mat",
            )
            for number in CWRU_FILES
        ),
        notes="12 kHz drive-end faults and 48 kHz normal baseline; 40 files.",
    ),
    "paderborn": DatasetSource(
        name="paderborn",
        title="Paderborn University bearing data (KAt-DataCenter)",
        kind="direct",
        role="source+target",
        licence="CC BY-NC 4.0",
        citation="C. Lessmeier et al., Condition monitoring of bearing damage in electromechanical "
        "drive systems by using motor current signals of electric motors, PHME 2016.",
        homepage="https://mb.uni-paderborn.de/kat/datacenter",
        files=tuple(
            RemoteFile(
                url=f"https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar",
                relpath=f"{code}.rar",
            )
            for code in PADERBORN_HEALTHY + PADERBORN_ARTIFICIAL + PADERBORN_REAL
        ),
        notes="Host refused connections from the development machine on 2026-10-25; "
        "download on Ruche or through a VPN. 32 RAR archives.",
    ),
    "xjtu_sy": DatasetSource(
        name="xjtu_sy",
        title="XJTU-SY rolling element bearing accelerated life test datasets",
        kind="gdrive",
        role="source",
        licence="Publicly available for research; no licence stated (cite the paper)",
        citation="B. Wang, Y. Lei, N. Li, N. Li, A hybrid prognostics approach for estimating "
        "remaining useful life of rolling element bearings, IEEE Trans. Reliability 69(1), "
        "401-412, 2020.",
        homepage="https://biaowang.tech/xjtu-sy-bearing-datasets/",
        files=tuple(
            RemoteFile(url=f"gdrive:{file_id}", relpath=name)
            for name, file_id in XJTU_SY_PARTS.items()
        ),
        notes="Six-part RAR archive on Google Drive; needs a RAR extractor that handles volumes.",
    ),
    "phm2010": DatasetSource(
        name="phm2010",
        title="PHM Society 2010 data challenge: CNC milling cutter wear",
        kind="manual",
        role="target",
        licence="No licence stated; PHM Society terms of use",
        citation="PHM Society 2010 Conference Data Challenge, "
        "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
        homepage="https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
        notes="The original CDN (cdn.teamholistic.net) now serves a parking page. Mirrors need "
        "an account: Hugging Face AI4Manufacturing/PHM2010 (gated), IEEE DataPort, Kaggle. "
        "Place c1.zip, c4.zip, c6.zip in the raw folder by hand.",
    ),
    "mbhm": DatasetSource(
        name="mbhm",
        title="MBHM: multimodal bearing health management corpus (BearLLM)",
        kind="huggingface",
        role="optional",
        licence="MIT",
        citation="H. Peng et al., BearLLM, AAAI 2025.",
        homepage="https://huggingface.co/datasets/SIA-IDE/MBHM",
        params={
            "repo": "SIA-IDE/MBHM",
            "files": ["README.md", "metadata.sqlite", "metadata.parquet", "corpus.json"],
            "large_files": ["data.hdf5"],
        },
        notes="data.hdf5 is 13 GB and is fetched only if the metadata check passes.",
    ),
}


def get_source(name: str) -> DatasetSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(SOURCES))}") from None
