"""Dataset readers: publisher files -> :class:`pdmfm.datasets.base.Recording`.

``get_reader(name)`` returns a reader with ``index()`` (cheap listing) and ``load(info)``.
Paderborn and PHM 2010 readers follow the published formats and are marked
``VERIFIED_ON_REAL_DATA = False`` until their files are downloaded.
"""

from __future__ import annotations

from pdmfm.datasets.base import ChannelData, Reader, Recording, RecordingInfo, iter_recordings


def get_reader(name: str, **kwargs) -> Reader:
    if name == "cwru":
        from pdmfm.datasets.cwru import CWRU as cls
    elif name == "kaist":
        from pdmfm.datasets.kaist import KAIST as cls
    elif name == "xjtu_sy":
        from pdmfm.datasets.xjtu_sy import XJTUSY as cls
    elif name == "nonastreda":
        from pdmfm.datasets.nonastreda import Nonastreda as cls
    elif name == "najafi_thermal":
        from pdmfm.datasets.najafi import Najafi as cls
    elif name == "paderborn":
        from pdmfm.datasets.paderborn import Paderborn as cls
    elif name == "phm2010":
        from pdmfm.datasets.phm2010 import PHM2010 as cls
    elif name == "nasa_milling":
        from pdmfm.datasets.nasa_milling import NASAMilling as cls
    else:
        raise KeyError(f"no reader for {name!r}")
    return cls(**kwargs)


__all__ = ["ChannelData", "Reader", "Recording", "RecordingInfo", "get_reader", "iter_recordings"]
