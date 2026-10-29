"""Evidential losses and combination rules (pdmfm.losses) against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from pdmfm import losses


def t(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


# ----------------------------------------------------------------------------- opinions


def test_opinion_beliefs_and_uncertainty_sum_to_one():
    evidence = torch.rand(20, 4, dtype=torch.float64) * 10
    belief, u = losses.opinion(evidence)
    torch.testing.assert_close(belief.sum(-1) + u, torch.ones(20, dtype=torch.float64))
    assert torch.all(u <= 1) and torch.all(u > 0)


def test_dempster_on_two_conflicting_opinions():
    # e1 = [2, 0]: S = 4, b = [0.5, 0], u = 0.5; e2 = [0, 2] mirrors it. Conflict
    # C = b1_0 b2_1 = 0.25; b_k = 0.25 / 0.75 = 1/3 each, u = 0.25 / 0.75 = 1/3, so S = 6
    # and the combined evidence is b S = [2, 2].
    combined, conflict = losses.ds_combine(t(2.0, 0.0), t(0.0, 2.0))
    assert float(conflict) == pytest.approx(0.25)
    torch.testing.assert_close(combined, t(2.0, 2.0))


def test_dempster_on_two_agreeing_opinions():
    # e1 = e2 = [2, 0]: C = 0; b_0 = 0.25 + 0.25 + 0.25 = 0.75, u = 0.25 -> S = 8, e = [6, 0].
    combined, conflict = losses.ds_combine(t(2.0, 0.0), t(2.0, 0.0))
    assert float(conflict) == pytest.approx(0.0)
    torch.testing.assert_close(combined, t(6.0, 0.0))
    _, u = losses.opinion(combined)
    assert float(u) < float(losses.opinion(t(2.0, 0.0))[1])  # agreement lowers uncertainty


def test_a_vacuous_opinion_is_the_identity():
    evidence = torch.rand(10, 5, dtype=torch.float64) * 7
    vacuous = torch.zeros_like(evidence)
    for combined, conflict in (
        losses.ds_combine(evidence, vacuous),
        losses.ds_combine(vacuous, evidence),
    ):
        torch.testing.assert_close(combined, evidence)
        torch.testing.assert_close(conflict, torch.zeros(10, dtype=torch.float64))


def test_dempster_is_commutative_and_associative():
    a, b, c = (torch.rand(8, 3, dtype=torch.float64) * 5 for _ in range(3))
    torch.testing.assert_close(losses.ds_combine(a, b)[0], losses.ds_combine(b, a)[0])
    left = losses.ds_combine(losses.ds_combine(a, b)[0], c)[0]
    right = losses.ds_combine(a, losses.ds_combine(b, c)[0])[0]
    torch.testing.assert_close(left, right)
    stacked = torch.stack([a, b, c], 1)
    torch.testing.assert_close(losses.ds_combine_all(stacked)[0], left)
    torch.testing.assert_close(losses.ds_combine_all(stacked[:, [2, 0, 1]])[0], left)


def test_combine_all_skips_missing_modalities_and_reports_conflicts():
    a, b = torch.rand(6, 4) * 5, torch.rand(6, 4) * 5
    stacked = torch.stack([a, torch.zeros_like(a), b], 1)
    combined, conflicts = losses.ds_combine_all(stacked)
    torch.testing.assert_close(combined, losses.ds_combine(a, b)[0])
    assert conflicts.shape == (6, 2)
    torch.testing.assert_close(conflicts[:, 0], torch.zeros(6))  # against the vacuous one
    one, none = losses.ds_combine_all(a.unsqueeze(1))
    torch.testing.assert_close(one, a)
    assert none.shape == (6, 0)


def test_combination_keeps_a_valid_opinion():
    combined, conflict = losses.ds_combine_all(torch.rand(50, 4, 6) * 20)
    belief, u = losses.opinion(combined)
    assert torch.all(combined >= 0) and torch.all((conflict >= 0) & (conflict < 1))
    torch.testing.assert_close(belief.sum(-1) + u, torch.ones(50))


# ----------------------------------------------------------------------------- TMC loss


def test_kl_to_uniform_is_zero_at_the_uniform_dirichlet_only():
    assert float(losses.kl_to_uniform(torch.ones(1, 4))) == pytest.approx(0.0, abs=1e-6)
    assert float(losses.kl_to_uniform(t(3.0, 1.0, 1.0))) > 0


def test_kl_matches_the_closed_form_for_two_classes():
    # Dir(a, 1) vs Dir(1, 1): Beta(a, 1) has density a x^(a-1); KL = log a - 1 + 1/a.
    a = 3.0
    expected = math.log(a) - 1 + 1 / a
    assert float(losses.kl_to_uniform(t(a, 1.0))) == pytest.approx(expected, rel=1e-6)


def test_tmc_loss_without_annealing_is_the_expected_cross_entropy():
    evidence, target = t(4.0, 1.0, 0.0), torch.tensor(0)
    alpha = evidence + 1
    expected = torch.digamma(alpha.sum()) - torch.digamma(alpha[0])
    torch.testing.assert_close(losses.tmc_loss(evidence, target, 0.0), expected)


def test_tmc_loss_rewards_true_evidence_and_penalizes_wrong_evidence():
    target = torch.tensor([0])
    base = float(losses.tmc_loss(t(1.0, 1.0, 1.0).unsqueeze(0), target, 1.0))
    more_true = float(losses.tmc_loss(t(9.0, 1.0, 1.0).unsqueeze(0), target, 1.0))
    more_wrong = float(losses.tmc_loss(t(1.0, 9.0, 1.0).unsqueeze(0), target, 1.0))
    assert more_true < base < more_wrong


def test_the_kl_term_ignores_the_true_class_evidence():
    target = torch.tensor([1])
    low = losses.tmc_loss(t(0.0, 1.0).unsqueeze(0), target, 1.0)
    high = losses.tmc_loss(t(0.0, 50.0).unsqueeze(0), target, 1.0)
    kl_low = low - losses.tmc_loss(t(0.0, 1.0).unsqueeze(0), target, 0.0)
    kl_high = high - losses.tmc_loss(t(0.0, 50.0).unsqueeze(0), target, 0.0)
    torch.testing.assert_close(kl_low, torch.zeros(1, dtype=torch.float64), atol=1e-7, rtol=0)
    torch.testing.assert_close(kl_high, torch.zeros(1, dtype=torch.float64), atol=1e-7, rtol=0)


def test_tmc_loss_gradients_are_finite_at_zero_evidence():
    evidence = torch.zeros(4, 3, requires_grad=True)
    losses.tmc_loss(evidence, torch.tensor([0, 1, 2, 0]), 1.0).sum().backward()
    assert evidence.grad is not None and torch.isfinite(evidence.grad).all()


def test_expected_probability_is_the_dirichlet_mean():
    torch.testing.assert_close(losses.expected_probability(t(3.0, 0.0)), t(0.8, 0.2))


# ----------------------------------------------------------------------------- NIG


def test_nig_parameters_respect_the_constraints():
    params = losses.nig_parameters(torch.randn(100, 4) * 10)
    _, nu, alpha, beta = params.unbind(-1)
    assert torch.all(nu > 0) and torch.all(alpha > 1) and torch.all(beta > 0)


@pytest.mark.parametrize(
    "gamma,nu,alpha,beta,y", [(0.0, 1.0, 2.0, 1.0, 0.5), (3.0, 0.2, 1.5, 4.0, -2.0)]
)
def test_nig_nll_is_the_student_t_log_density(gamma, nu, alpha, beta, y):
    params = t(gamma, nu, alpha, beta)
    dof, scale = losses.nig_predictive_scale(params)
    expected = -stats.t.logpdf(y, df=float(dof), loc=gamma, scale=float(scale))
    assert float(losses.nig_nll(params, t(y))) == pytest.approx(expected, rel=1e-8)


def test_nig_loss_adds_the_evidence_regularizer():
    params, y = t(1.0, 2.0, 3.0, 0.5), t(4.0)
    expected = losses.nig_nll(params, y) + 0.1 * 3.0 * (2 * 2.0 + 3.0)
    torch.testing.assert_close(losses.nig_loss(params, y, regularizer=0.1), expected)


def test_nig_uncertainties():
    aleatoric, epistemic = losses.nig_uncertainty(t(0.0, 2.0, 3.0, 4.0))
    assert float(aleatoric) == pytest.approx(2.0) and float(epistemic) == pytest.approx(1.0)


def test_fusing_two_identical_nigs_doubles_the_evidence():
    gamma, nu, alpha, beta = 1.5, 2.0, 3.0, 0.7
    fused = losses.nig_fuse(t(gamma, nu, alpha, beta), t(gamma, nu, alpha, beta))
    torch.testing.assert_close(fused, t(gamma, 2 * nu, 2 * alpha + 0.5, 2 * beta))


def test_monig_mean_is_the_nu_weighted_mean_and_disagreement_raises_beta():
    p1, p2 = t(0.0, 3.0, 2.0, 1.0), t(4.0, 1.0, 2.0, 1.0)
    fused = losses.nig_fuse(p1, p2)
    assert float(fused[0]) == pytest.approx(1.0)  # (3 * 0 + 1 * 4) / 4
    # beta = 1 + 1 + 3 (0 - 1)^2 / 2 + 1 (4 - 1)^2 / 2 = 2 + 1.5 + 4.5
    assert float(fused[3]) == pytest.approx(8.0)
    torch.testing.assert_close(fused, losses.nig_fuse(p2, p1))


def test_nig_fusion_skips_absent_modalities():
    params = losses.nig_parameters(torch.randn(5, 3, 4, dtype=torch.float64))
    present = torch.tensor([[True, False, True]] * 5)
    expected = losses.nig_fuse(params[:, 0], params[:, 2])
    torch.testing.assert_close(losses.nig_fuse_all(params, present), expected)
    only_middle = torch.tensor([[False, True, False]] * 5)
    torch.testing.assert_close(losses.nig_fuse_all(params, only_middle), params[:, 1])
    none = torch.zeros(5, 3, dtype=torch.bool)
    torch.testing.assert_close(losses.nig_fuse_all(params, none), params[:, 0])


def test_a_fitted_nig_gives_calibrated_intervals():
    """Fit one NIG to samples of N(3, 2^2) by the loss; its 90 % Student-t interval then
    covers about 90 % of fresh samples."""
    torch.manual_seed(0)
    y = 3.0 + 2.0 * torch.randn(4000, dtype=torch.float64)
    raw = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.05)
    for _ in range(1500):
        optimizer.zero_grad()
        loss = losses.nig_nll(losses.nig_parameters(raw).expand(len(y), 4), y).mean()
        loss.backward()
        optimizer.step()
    params = losses.nig_parameters(raw.detach())
    dof, scale = losses.nig_predictive_scale(params)
    assert float(params[0]) == pytest.approx(3.0, abs=0.1)
    fresh = 3.0 + 2.0 * np.random.default_rng(1).standard_normal(20_000)
    half = stats.t.ppf(0.95, float(dof)) * float(scale)
    coverage = np.mean(np.abs(fresh - float(params[0])) <= half)
    assert coverage == pytest.approx(0.9, abs=0.02)
