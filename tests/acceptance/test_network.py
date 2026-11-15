"""Live checks of the stable data sources (run with PDMFM_NETWORK_TESTS=1; weekly in CI).

Sizes and hashes are the ones recorded when the data was downloaded on 2026-10-25; a change
means the publisher altered the files, which would invalidate earlier results.
"""

from __future__ import annotations

import pytest
import requests

from pdmfm import fetch
from pdmfm.sources import get_source

pytestmark = pytest.mark.network


def test_mendeley_listing_matches_the_downloaded_najafi_files():
    files = {f.relpath: f for f in fetch.list_remote_files(get_source("najafi_thermal"))}
    assert files["IR-Motor-bmp.rar"].size == 9_577_478
    assert files["IR-Motor-bmp.rar"].sha256 == (
        "443815d41f96baf971008687eb2a80e7bc21ca7bfc3dc61cb8fa7ad9295020ea"
    )


def test_kaist_archives_are_unchanged():
    files = {f.relpath: f for f in fetch.list_remote_files(get_source("kaist"))}
    assert files["vibration.zip"].size == 2_659_778_162
    assert files["current,temp.zip"].size == 1_547_790_068


def test_mbhm_metadata_is_listed():
    files = {f.relpath: f for f in fetch.list_remote_files(get_source("mbhm"))}
    assert files["metadata.sqlite"].size == 4_464_640


@pytest.mark.parametrize(
    "url, size",
    [
        ("https://engineering.case.edu/sites/default/files/97.mat", 3_903_344),
        ("https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip", 14_731_306),
    ],
)
def test_direct_downloads_keep_their_size(url, size):
    response = requests.head(
        url, headers={"User-Agent": fetch.USER_AGENT}, timeout=30, allow_redirects=True
    )
    assert response.status_code == 200
    assert int(response.headers["Content-Length"]) == size
