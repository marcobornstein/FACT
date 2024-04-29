import numpy as np


def agent_contribution(cost, offset=1):
    optimal_data = int(np.sqrt(offset / cost))
    data_cost = cost * optimal_data
    return optimal_data, data_cost


def agent_loss(num_data, cost, offset=1):
    return (offset / num_data) + (num_data * cost)


def sandwich_game(random, reported_cost, random_costs):
    if random:
        baseline_costs = np.sort(np.random.choice(random_costs, 2, replace=False))
        # print(baseline_costs)
        baseline_cost_l = baseline_costs[0]
        baseline_cost_h = baseline_costs[1]
    else:
        baseline_cost_l = np.quantile(random_costs, 1 / 3)
        baseline_cost_h = np.quantile(random_costs, 2 / 3)

        # print(np.array([baseline_cost_l, baseline_cost_h]))

    lower_boolean = baseline_cost_l < reported_cost
    upper_boolean = baseline_cost_h > reported_cost
    return lower_boolean * upper_boolean, 3


def cost_deflation_game(random, reported_cost, random_costs):
    baseline_cost = np.random.choice(random_costs) if random else np.median(random_costs)
    return reported_cost < baseline_cost, 2


def truthfulness_mechanism(true_cost, num_data, agent_net_loss, average_agent_net_loss,
                           agents=1000, rounds=100000, h=81, random=True, sandwich=True):

    benefit_tracker = np.zeros(shape=(rounds, h))
    cost_std = true_cost / 10
    cost_mean = true_cost

    # multiple rounds for simulation
    for r in range(rounds):
        # loop over all possible costs
        epsilons = np.linspace(-0.2, 0.2, h, endpoint=True)

        # new reported costs
        reported_costs = true_cost * (1 + epsilons)

        # benefit of lying about cost
        data_cost_net = num_data * (true_cost - reported_costs)

        # simulate random agents and their costs from normal distribution
        random_costs = np.random.normal(cost_mean, cost_std, size=(agents,))

        booleans, multiplier = sandwich_game(random, reported_costs, random_costs) if sandwich else (
            cost_deflation_game(random, reported_costs, random_costs))

        benefit_tracker[r, :] = data_cost_net + agent_net_loss - (multiplier * booleans * average_agent_net_loss)

    return benefit_tracker
