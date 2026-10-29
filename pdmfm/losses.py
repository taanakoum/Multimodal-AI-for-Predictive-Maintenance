"""Evidential losses and combination rules (plan, "Model architecture" and "Trust").

Rewritten from the papers, whose repositories carry no licence:

* **Classes (TMC; Han et al., ICLR 2021 / TPAMI 2023).** Evidence e >= 0 gives a Dirichlet
  alpha = e + 1, with S = sum(alpha), belief b_k = e_k / S and uncertainty u = K / S. Two
  opinions combine by Dempster's rule with conflict C = sum_{i != j} b1_i b2_j:
  b = (b1 b2 + b1 u2 + b2 u1) / (1 - C), u = u1 u2 / (1 - C). A vacuous opinion (e = 0, u = 1)
  is the identity, so a missing modality simply has zero evidence. The loss is the expected
  cross-entropy under the Dirichlet plus an annealed KL to Dir(1) of the evidence on wrong
  classes.
* **Regression (deep evidential regression; Amini et al., NeurIPS 2020).** A Normal-Inverse-
  Gamma (gamma, nu, alpha, beta) per target; loss = NIG negative log-likelihood +
  lambda |y - gamma| (2 nu + alpha). Aleatoric variance beta / (alpha - 1), epistemic
  beta / (nu (alpha - 1)); the predictive is a Student-t with 2 alpha degrees of freedom.
* **Regression fusion (MoNIG; Ma et al., NeurIPS 2021):** the NIG summation
  nu = nu1 + nu2, gamma = (nu1 gamma1 + nu2 gamma2) / nu, alpha = alpha1 + alpha2 + 1/2,
  beta = beta1 + beta2 + nu1 (gamma1 - gamma)^2 / 2 + nu2 (gamma2 - gamma)^2 / 2.
"""

from __future__ import annotations

import math

import torch

# ----------------------------------------------------------------------------- classes


def opinion(evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(belief (..., K), uncertainty (...)) of evidence (..., K)."""
    k = evidence.shape[-1]
    strength = evidence.sum(-1, keepdim=True) + k
    return evidence / strength, (k / strength).squeeze(-1)


def ds_combine(e1: torch.Tensor, e2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dempster's rule on two evidence tensors (..., K). Returns (combined evidence, conflict C)."""
    k = e1.shape[-1]
    b1, u1 = opinion(e1)
    b2, u2 = opinion(e2)
    agreement = (b1 * b2).sum(-1)
    conflict = b1.sum(-1) * b2.sum(-1) - agreement  # sum over i != j of b1_i b2_j
    scale = 1.0 / (1.0 - conflict).clamp_min(1e-8)
    belief = (b1 * b2 + b1 * u2.unsqueeze(-1) + b2 * u1.unsqueeze(-1)) * scale.unsqueeze(-1)
    uncertainty = (u1 * u2 * scale).clamp_min(1e-8)
    strength = k / uncertainty
    return belief * strength.unsqueeze(-1), conflict


def ds_combine_all(evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine M opinions (..., M, K) in turn. Returns (evidence (..., K), conflicts (..., M-1)).
    A modality with zero evidence (missing) leaves the result unchanged."""
    combined = evidence[..., 0, :]
    conflicts = []
    for m in range(1, evidence.shape[-2]):
        combined, conflict = ds_combine(combined, evidence[..., m, :])
        conflicts.append(conflict)
    stacked = (
        torch.stack(conflicts, -1) if conflicts else combined.new_zeros(combined.shape[:-1] + (0,))
    )
    return combined, stacked


def kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1)) per row."""
    k = alpha.shape[-1]
    strength = alpha.sum(-1)
    return (
        torch.lgamma(strength)
        - math.lgamma(k)
        - torch.lgamma(alpha).sum(-1)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(strength).unsqueeze(-1))).sum(-1)
    )


def tmc_loss(evidence: torch.Tensor, target: torch.Tensor, annealing: float) -> torch.Tensor:
    """Per-row TMC loss: expected cross-entropy under Dir(e + 1) plus ``annealing`` times the
    KL of the wrong-class evidence to Dir(1). ``target`` holds class indices."""
    alpha = evidence + 1.0
    one_hot = torch.nn.functional.one_hot(target, alpha.shape[-1]).to(alpha.dtype)
    strength = alpha.sum(-1, keepdim=True)
    expected_ce = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(-1)
    wrong = one_hot + (1.0 - one_hot) * alpha  # alpha with the true class's evidence removed
    return expected_ce + annealing * kl_to_uniform(wrong)


def expected_probability(evidence: torch.Tensor) -> torch.Tensor:
    """Mean of Dir(e + 1): the class probabilities."""
    alpha = evidence + 1.0
    return alpha / alpha.sum(-1, keepdim=True)


# ----------------------------------------------------------------------------- regression


def nig_parameters(raw: torch.Tensor, min_evidence: float = 1e-4) -> torch.Tensor:
    """(..., 4) unconstrained outputs -> (gamma, nu > 0, alpha > 1, beta > 0)."""
    gamma, nu, alpha, beta = raw.unbind(-1)
    softplus = torch.nn.functional.softplus
    return torch.stack(
        [
            gamma,
            softplus(nu) + min_evidence,
            softplus(alpha) + 1.0 + min_evidence,
            softplus(beta) + min_evidence,
        ],
        -1,
    )


def nig_nll(params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of y under the NIG's Student-t predictive (Amini et al., eq. 8)."""
    gamma, nu, alpha, beta = params.unbind(-1)
    omega = 2.0 * beta * (1.0 + nu)
    return (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def nig_loss(params: torch.Tensor, y: torch.Tensor, regularizer: float = 0.01) -> torch.Tensor:
    """Per-row NIG loss: NLL + lambda |y - gamma| (2 nu + alpha)."""
    gamma, nu, alpha, _ = params.unbind(-1)
    return nig_nll(params, y) + regularizer * (y - gamma).abs() * (2.0 * nu + alpha)


def nig_fuse(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """MoNIG summation of two NIG parameter tensors (..., 4)."""
    g1, n1, a1, b1 = p1.unbind(-1)
    g2, n2, a2, b2 = p2.unbind(-1)
    nu = n1 + n2
    gamma = (n1 * g1 + n2 * g2) / nu
    alpha = a1 + a2 + 0.5
    beta = b1 + b2 + 0.5 * n1 * (g1 - gamma) ** 2 + 0.5 * n2 * (g2 - gamma) ** 2
    return torch.stack([gamma, nu, alpha, beta], -1)


def nig_fuse_all(params: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Fuse M NIGs (..., M, 4) in turn, skipping absent ones (present (..., M) bool). A
    sample with no present modality keeps the first NIG (the caller masks it out)."""
    fused = params[..., 0, :]
    have = present[..., 0]
    for m in range(1, params.shape[-2]):
        current = params[..., m, :]
        merged = nig_fuse(fused, current)
        use = present[..., m]
        fused = torch.where(
            (use & have).unsqueeze(-1), merged, torch.where(use.unsqueeze(-1), current, fused)
        )
        have = have | use
    return fused


def nig_uncertainty(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(aleatoric variance beta/(alpha-1), epistemic variance beta/(nu (alpha-1)))."""
    _, nu, alpha, beta = params.unbind(-1)
    return beta / (alpha - 1.0), beta / (nu * (alpha - 1.0))


def nig_predictive_scale(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(degrees of freedom 2 alpha, scale) of the Student-t predictive."""
    _, nu, alpha, beta = params.unbind(-1)
    return 2.0 * alpha, torch.sqrt(beta * (1.0 + nu) / (nu * alpha))
