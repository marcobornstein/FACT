import numpy as np


def agent_contribution(cost, offset=2):
    optimal_data = int(np.sqrt(offset / (2 * cost)))
    data_cost = cost * optimal_data
    return optimal_data, data_cost


def agent_loss(num_data, cost, offset=2):
    return (offset / (2 * num_data)) + (num_data * cost)


def sandwich_game(reported_cost, random_costs, baseline_costs=None):
    if baseline_costs is None:
        baseline_costs = np.sort(np.random.choice(random_costs, 2, replace=False))
    lower_boolean = baseline_costs[0] < reported_cost
    upper_boolean = reported_cost < baseline_costs[1]
    return lower_boolean * upper_boolean, 3


def cost_deflation_game(reported_cost, random_costs, baseline_cost=None):
    if baseline_cost is None:
        baseline_cost = np.random.choice(random_costs)
    return reported_cost < baseline_cost, 2


def truthfulness_mechanism(true_cost, num_data, agent_net_loss, other_agent_avg_net_loss, num_agents,
                           agents=1000, rounds=20000, h=81, normal=True, sandwich=True):

    benefit_random = np.zeros(shape=(rounds, h))
    benefit_det = np.zeros(shape=(rounds, h))
    cost_std = true_cost / 10
    cost_mean = true_cost
    fact_reward = other_agent_avg_net_loss * (num_agents - 1) / num_agents

    # change in percentage of costs
    epsilons = np.linspace(-0.2, 0.2, h, endpoint=True)

    # new reported costs
    reported_costs = true_cost * (1 + epsilons)

    # benefit of lying about cost
    data_cost_net = num_data * (true_cost - reported_costs)

    # multiple rounds for random simulation
    for r in range(rounds):

        # simulate random agents and their costs from normal distribution
        random_costs = np.random.normal(cost_mean, cost_std, size=(agents,)) if normal \
            else np.random.uniform(0, cost_mean * 2, size=(agents,))

        # deterministic baseline costs
        deterministic_bc = [np.quantile(random_costs, 1 / 3), np.quantile(random_costs, 2 / 3)] if sandwich \
            else np.median(random_costs)

        # get deterministic result
        booleans_det, multiplier_det = sandwich_game(reported_costs, random_costs, deterministic_bc) if sandwich else (
                cost_deflation_game(reported_costs, random_costs, deterministic_bc))
        benefit_det[r, :] = data_cost_net + agent_net_loss - (multiplier_det * booleans_det * fact_reward)

        booleans_random, multiplier_random = sandwich_game(reported_costs, random_costs) if sandwich else (
            cost_deflation_game(reported_costs, random_costs))
        # compute reward from FACT for given round
        benefit_random[r, :] = data_cost_net + agent_net_loss - (multiplier_random * booleans_random * fact_reward)

    # find average benefit
    avg_benefit_random = np.mean(benefit_random, axis=0)
    avg_benefit_det = np.mean(benefit_det, axis=0)

    # determine average loss for FACT over the three randomly selected devices
    three_agent_net_loss = agent_net_loss + 2 * other_agent_avg_net_loss
    three_agent_fact_rewards = fact_reward + 2 * ((other_agent_avg_net_loss * (num_agents - 2) / num_agents)
                                                  + agent_net_loss / num_agents)
    fact_loss = three_agent_net_loss - three_agent_fact_rewards

    return fact_loss, avg_benefit_random, avg_benefit_det
