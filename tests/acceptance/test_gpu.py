"""CUDA acceptance: numerical agreement and exact stochastic checkpoint continuation."""

import pytest
import torch

from pdmfm.compute import autocast
from pdmfm.losses import nig_loss, nig_parameters

pytestmark = pytest.mark.gpu


def test_cuda_rng_round_trip_and_evidential_gradients():
    torch.manual_seed(7)
    state = torch.cuda.get_rng_state_all()
    expected = torch.rand(32, device="cuda")
    torch.cuda.set_rng_state_all(state)
    torch.testing.assert_close(torch.rand(32, device="cuda"), expected, rtol=0, atol=0)
    model = torch.nn.Linear(8, 4).cuda()
    x = torch.ones(16, 8, device="cuda")
    reference = nig_parameters(model(x))
    with autocast("cuda", "auto"):
        actual = nig_parameters(model(x))
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.03)
    nig_loss(actual, torch.zeros(16, device="cuda")).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_cuda_training_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import numpy as np

    from pdmfm import protocols
    from pdmfm.model.inputs import assemble
    from pdmfm.train import TrainConfig, fit
    from tests.support.synthetic import build_planb_root

    monkeypatch.setenv("PDMFM_DATA_ROOT", str(tmp_path))
    build_planb_root(tmp_path, datasets=("nonastreda",))
    inputs = assemble("nonastreda", "stats", ("F",))
    split = protocols.nonastreda_fold(inputs.manifest, 1)
    train, val = np.flatnonzero(split.train), np.flatnonzero(split.val)
    cfg = TrainConfig(epochs=3, d=16, device="cuda", precision="fp32", input_dropout=0.3)
    torch.use_deterministic_algorithms(True)
    try:
        expected = fit(inputs, train, val, cfg, 11, log=lambda *_: None)

        def interrupt(message):
            if "checkpoint epoch 1/3" in message:
                raise InterruptedError

        checkpoint = tmp_path / "last.pt"
        with pytest.raises(InterruptedError):
            fit(inputs, train, val, cfg, 11, log=interrupt, checkpoint=checkpoint)
        actual = fit(inputs, train, val, cfg, 11, checkpoint=checkpoint)
        for name, value in expected.model.state_dict().items():
            torch.testing.assert_close(actual.model.state_dict()[name], value, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(False)
