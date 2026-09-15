import numpy as np
import pytest
from scipy.stats import norm

from fact.config import DATASETS
from fact.mechanism import (
    cost_deviations,
    expected_truthfulness_benefit,
    fact_loss,
    free_riding_penalty,
    optimal_contribution,
)


@pytest.mark.parametrize("dataset, expected", [("cifar10", 3125), ("mnist", 3750), ("ham10000", 801)])
def test_optimal_contribution_matches_paper(dataset, expected):
    config = DATASETS[dataset]
    assert optimal_contribution(config["marginal_cost"], config["gamma_sigma_l"]) == expected


@pytest.mark.parametrize("dataset", ["cifar10", "mnist", "ham10000"])
def test_penalty_plus_data_cost_is_minimised_at_the_optimum(dataset):
    """Theorem 3: the free-riding penalty moves the federated optimum back to m*."""
    config = DATASETS[dataset]
    cost, gamma_sigma_l = config["marginal_cost"], config["gamma_sigma_l"]
    m_star = optimal_contribution(cost, gamma_sigma_l)
    m = np.arange(m_star - 500, m_star + 501)
    total_cost = free_riding_penalty(m, cost, m_star, m_star * config["num_agents"], gamma_sigma_l) + m * cost
    assert m[np.argmin(total_cost)] == m_star


def test_fact_loss_subtracts_the_expected_reward():
    assert fact_loss(2.0, 1.0, num_agents=4) == pytest.approx(1.25)


def test_truthfulness_benefit_matches_lemma_2():
    """Win probability is 2F(1 - F); a truthful agent (F = 1/2) expects exactly the reward."""
    cost, others, num_agents, h, rounds = 1e-7, 0.8, 16, 41, 400_000
    benefit = expected_truthfulness_benefit(cost, others, num_agents, h=h, rounds=rounds, rng=np.random.default_rng(0))

    win = 2 * norm.cdf(cost_deviations(h) / 0.1) * (1 - norm.cdf(cost_deviations(h) / 0.1))
    reward = others * (num_agents - 1) / num_agents
    std_error = 2 * reward * np.sqrt(win * (1 - win) / rounds)
    np.testing.assert_array_less(np.abs(benefit - 2 * reward * win), 5 * std_error + 1e-12)
    assert abs(np.argmax(benefit) - h // 2) <= 1


def test_truthfulness_benefit_does_not_depend_on_chunking():
    kwargs = {"true_cost": 1.0, "others_net_improvement": 1.0, "num_agents": 3, "h": 11, "rounds": 1003}
    small = expected_truthfulness_benefit(**kwargs, rng=np.random.default_rng(1), chunk_size=7)
    large = expected_truthfulness_benefit(**kwargs, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(small, large)
