"""The FACT mechanism: optimal contributions, the free-riding penalty and the sandwich game.

Equation and theorem numbers refer to the NeurIPS 2024 paper.
"""

import numpy as np

# The server's alpha in [0, 2) (Lemma 1). Values close to 2 make the penalty at the optimum vanish.
ALPHA = 2 - 1e-6

# Reported costs are swept over c * (1 + eps) for eps in [-MAX_COST_DEVIATION, MAX_COST_DEVIATION].
MAX_COST_DEVIATION = 0.2


def optimal_contribution(cost, gamma_sigma_l):
    """Locally optimal number of samples m* = sqrt(gamma_sigma_l / 2c) (Theorem 1)."""
    return int(np.sqrt(gamma_sigma_l / (2 * cost)))


def penalty_scale(cost, m_star, total_data, gamma_sigma_l, alpha=ALPHA):
    """Penalty harshness lambda_i that keeps PFL individually rational (Eq. 21)."""
    coefficient = m_star * total_data / ((2 - alpha) * gamma_sigma_l * (total_data - m_star))
    return coefficient * (cost - gamma_sigma_l / (2 * total_data**2)) ** 2


def free_riding_penalty(m, cost, m_star, total_data, gamma_sigma_l, alpha=ALPHA):
    """Free-riding penalty P_fr(m) paid for contributing m instead of m* samples (Eq. 4)."""
    lam = penalty_scale(cost, m_star, total_data, gamma_sigma_l, alpha)
    return lam * (cost / (2 * lam) - gamma_sigma_l / (4 * lam * total_data**2) + m_star - m) ** 2


def fact_loss(local_loss, others_net_improvement, num_agents):
    """Expected loss of an agent in FACT.

    The agent pays its own improvement up front and, by winning the sandwich game with
    probability 1/3, receives three times the average improvement of the others (Eq. 10).
    """
    return local_loss - others_net_improvement * (num_agents - 1) / num_agents


def cost_deviations(h):
    """Relative misreports eps on which the truthfulness curve is evaluated."""
    return np.linspace(-MAX_COST_DEVIATION, MAX_COST_DEVIATION, h)


def expected_truthfulness_benefit(true_cost, others_net_improvement, num_agents, h=81, rounds=100_000,
                                  rng=None, chunk_size=10_000):
    """Monte Carlo estimate of the expected improvement in loss when reporting c * (1 + eps).

    Each round draws the two opposing costs C_a, C_b ~ N(c, (c/10)^2) and the agent wins if its
    report lies strictly between them (Eq. 10). The paper draws the two opponents from a pool of
    2,000 synthetic agents with those costs, which is the same distribution. The result is scaled
    so that a truthful agent, who wins half the time (Lemma 2), expects exactly the FACT reward.
    """
    rng = np.random.default_rng() if rng is None else rng
    reported = true_cost * (1 + cost_deviations(h))
    wins = np.zeros(h)
    for start in range(0, rounds, chunk_size):
        n = min(chunk_size, rounds - start)
        low, high = np.sort(rng.normal(true_cost, true_cost / 10, size=(n, 2)), axis=1).T
        wins += ((low[:, None] < reported) & (reported < high[:, None])).sum(axis=0)
    reward = others_net_improvement * (num_agents - 1) / num_agents
    return 2 * reward * wins / rounds
