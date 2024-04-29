import numpy as np


def agent_contribution(cost, offset=1):
    optimal_data = int(np.sqrt(offset / cost))
    data_cost = cost * optimal_data
    return optimal_data, data_cost


def agent_loss(num_data, cost, offset=1):
    return (offset / num_data) + (num_data * cost)


def agent_gain_inflation(true_cost, reported_cost, num_data, agent_net_loss, average_agent_net_loss,
                         agents=1000, random=False):

    # goal: show that in practice (not theoretical) if an agent trains locally on their optimal amount of data,
    # but trains in federated manner when lying about cost, it'll be worse off. It's best to be honest.

    # benefit of lying about cost
    data_cost_net = num_data * (reported_cost - true_cost)

    # simulate random agents and their costs from normal distribution
    random_costs = np.random.normal(true_cost, true_cost / 10, size=(agents,))

    # mechanism game
    if random:
        baseline_cost = np.random.choice(random_costs)
    else:
        baseline_cost = np.median(random_costs)

    if reported_cost < baseline_cost:
        return data_cost_net + agent_net_loss - 2 * average_agent_net_loss
    else:
        return data_cost_net + agent_net_loss


def agent_gain_truth(true_cost, reported_cost, num_data, agent_net_loss, average_agent_net_loss,
                     agents=1000, random=False):

    # goal: show that in practice (not theoretical) if an agent trains locally on their optimal amount of data,
    # but trains in federated manner when lying about cost, it'll be worse off. It's best to be honest.

    # benefit of lying about cost
    data_cost_net = num_data * (reported_cost - true_cost)

    # simulate random agents and their costs from normal distribution
    random_costs = np.random.normal(true_cost, true_cost / 10, size=(agents,))

    # mechanism game
    if random:
        baseline_costs = np.sort(np.random.choice(random_costs, 2, replace=False))
        baseline_cost_l = baseline_costs[0]
        baseline_cost_h = baseline_costs[1]
    else:
        baseline_cost_l = np.quantile(random_costs, 1 / 3)
        baseline_cost_h = np.quantile(random_costs, 2 / 3)

    print('====')
    print(baseline_cost_l)
    print(baseline_cost_h)
    print('====')

    if baseline_cost_l < reported_cost < baseline_cost_h:
        return data_cost_net + agent_net_loss - 3 * average_agent_net_loss
    else:
        return data_cost_net + agent_net_loss
