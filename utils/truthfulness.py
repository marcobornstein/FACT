import numpy as np


def agent_contribution(cost, offset=2):
    optimal_data = int(np.sqrt(offset / (2 * cost)))
    data_cost = cost * optimal_data
    return optimal_data, data_cost


def agent_loss(num_data, cost, offset=2):
    return (offset / (2 * num_data)) + (num_data * cost)


def sandwich_game(reported_cost, random_costs):
    baseline_costs = np.sort(np.random.choice(random_costs, 2, replace=False))
    lower_boolean = baseline_costs[0] < reported_cost
    upper_boolean = reported_cost < baseline_costs[1]
    return lower_boolean * upper_boolean, 3


def truthfulness_mechanism(true_cost, num_data, local_loss, agent_net_loss, other_agent_avg_net_loss, num_agents,
                           agents=1000, rounds=20000, h=81, normal=True):

    benefit_random = np.zeros(shape=(rounds, h))
    cost_std = true_cost / 10
    cost_mean = true_cost
    fact_reward = other_agent_avg_net_loss * (num_agents - 1) / num_agents

    # change in percentage of costs
    epsilons = np.linspace(-0.2, 0.2, h, endpoint=True)

    # new reported costs
    reported_costs = true_cost * (1 + epsilons)

    # benefit of lying about cost
    # data_cost_net = num_data * (true_cost - reported_costs)

    # multiple rounds for random simulation
    for r in range(rounds):

        # simulate random agents and their costs from normal distribution
        random_costs = np.random.normal(cost_mean, cost_std, size=(agents,)) if normal \
            else np.random.uniform(0, cost_mean * 2, size=(agents,))

        # sandwich competition
        booleans_random, multiplier_random = sandwich_game(reported_costs, random_costs)

        # compute reward from FACT for given round
        # benefit_random[r, :] = data_cost_net + local_loss - (multiplier_random * booleans_random * fact_reward)  # this is what the device thinks
        benefit_random[r, :] = 2 * booleans_random * fact_reward  # this is the expected improvement (thinks it wins 50% of time, so divide by three and multiple by 2)

    # find average benefit
    avg_benefit_random = np.mean(benefit_random, axis=0)

    # fact loss
    fact_loss = local_loss - (other_agent_avg_net_loss * (num_agents - 1) / num_agents)

    """
    # determine average loss for FACT over the three randomly selected devices (what's the expected gain)
    three_agent_net_loss = agent_net_loss + 2 * other_agent_avg_net_loss
    three_agent_fact_rewards = fact_reward + 2 * ((other_agent_avg_net_loss * (num_agents - 2) / num_agents)
                                                  + agent_net_loss / num_agents)
    fact_loss = three_agent_net_loss - three_agent_fact_rewards
    """

    return fact_loss, avg_benefit_random
