"""Regenerate the figures from the per-agent logs in output/.

    python plot.py                        # every figure
    python plot.py --only truthful loss   # selected families
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fact.config import DATASETS
from fact.mechanism import (
    cost_deviations,
    expected_truthfulness_benefit,
    fact_loss,
    free_riding_penalty,
    optimal_contribution,
)


def read_logs(folder, series, num_agents):
    """Stack r<i>-<series>.log of every agent into an array of shape (rows, agents)."""
    return np.column_stack([np.atleast_1d(np.loadtxt(os.path.join(folder, f"r{i}-{series}.log")))
                            for i in range(num_agents)])


class Experiment:
    """Repeated runs of one configuration, each run a folder of per-agent logs."""

    def __init__(self, results_dir, dataset, label, folder_template, runs=3):
        self.dataset = dataset
        self.label = label
        self.config = DATASETS[dataset]
        self.folders = [os.path.join(results_dir, self.config["output_group"], folder_template.format(run))
                        for run in range(1, runs + 1)]
        self.num_agents = int(re.search(r"-(\d+)devices$", folder_template).group(1))

    def curves(self, metric):
        """Per-run test curves (runs x epochs): local training averaged over agents, and the federated model."""
        local = np.stack([read_logs(f, f"local-epoch-{metric}", self.num_agents).mean(axis=1) for f in self.folders])
        fed = np.stack([read_logs(f, f"fed-epoch-{metric}", self.num_agents)[:, 0] for f in self.folders])
        return local, fed

    def mechanism(self):
        """Marginal cost, optimal contribution, penalty at the optimum and each agent's FACT reward input."""
        cost = read_logs(self.folders[0], "marginal-costs", self.num_agents)[0, 0]
        gamma_sigma_l = self.config["gamma_sigma_l"]
        m_star = optimal_contribution(cost, gamma_sigma_l)
        total = m_star * self.num_agents
        penalty = free_riding_penalty(m_star, cost, m_star, total, gamma_sigma_l)
        others = np.mean([read_logs(f, "benefits", self.num_agents)[1] for f in self.folders], axis=0)
        return cost, m_star, total, penalty, others


def bootstrap_band(runs, rng, group_size=30, groups=1000, percentiles=(1, 99)):
    """Percentile band of bootstrapped means for every column of `runs`."""
    means = runs[rng.integers(0, len(runs), size=(groups, group_size))].mean(axis=1)
    return np.percentile(means, percentiles, axis=0)


CURVE_LIMITS = {
    ("mnist", "loss"): ([0.01, 2.5], "log"),
    ("cifar10", "loss"): ([0, 125], "symlog"),
    ("mnist", "acc-top1"): ([0.75, 1.0], None),
    ("cifar10", "acc-top1"): ([0.1, 0.8], None),
}


def plot_curves(exp, metric, rng):
    local, fed = exp.curves(metric)
    epochs = np.arange(1, local.shape[1] + 1)
    for runs, color, label in ((local, "r", "Local Training"), (fed, "b", "Federated Training")):
        plt.plot(epochs, runs.mean(axis=0), color=color, label=label)
        plt.fill_between(epochs, *bootstrap_band(runs, rng), alpha=0.2, color=color)
    plt.xlabel("Epochs", fontsize=20, weight="bold")
    plt.ylabel("Test Loss" if metric == "loss" else "Test Accuracy", fontsize=20, weight="bold")
    plt.xticks(fontsize=17, weight="bold")
    plt.yticks(fontsize=17, weight="bold")
    plt.legend(loc="best", fontsize=15)
    if (exp.dataset, metric) in CURVE_LIMITS:
        ylim, yscale = CURVE_LIMITS[exp.dataset, metric]
        plt.ylim(ylim)
        if yscale:
            plt.yscale(yscale)
    plt.xlim([1, epochs[-1]])
    plt.grid(alpha=0.25)


def plot_loss_histogram(exp):
    local, fed = exp.curves("loss")
    local_loss, fed_loss = local[:, -1].mean(), fed[:, -1].mean()
    _, _, _, penalty, others = exp.mechanism()
    fact = fact_loss(local_loss, others, exp.num_agents).mean() + penalty
    plt.bar(["Local Training", "FACT Training", "Traditional FL"], [local_loss, fact, fed_loss],
            color=["tab:red", "tab:blue", "tab:green"])
    plt.ylabel("Expected Loss", fontsize=25, weight="bold")
    plt.xticks(fontsize=17, weight="bold")
    plt.yticks(fontsize=17, weight="bold")
    if exp.dataset == "cifar10":
        plt.ylim([0, 7])
    elif exp.dataset == "mnist":
        plt.ylim([0.001, 2])
        plt.yscale("log")
    plt.grid(alpha=0.25, axis="y")
    plt.tick_params(axis="both", which="major", labelsize=16)


def plot_truthfulness(experiments, rng, h=121):
    styles = [(":", " IID"), ("--", " N-IID (D-0.6)"), ("-", " N-IID (D-0.3)")]
    for exp, (linestyle, label) in zip(experiments, styles):
        cost, _, _, penalty, others = exp.mechanism()
        # Averaging the agents' curves equals one curve for the mean reward; simulate as many rounds as all agents.
        benefit = expected_truthfulness_benefit(cost, others.mean(), exp.num_agents, h=h,
                                                rounds=100_000 * exp.num_agents, rng=rng)
        plt.plot(100 * cost_deviations(h), benefit + penalty, "r", linestyle=linestyle, label=label)
    plt.xlabel("Percent (%) Added/Subtracted from True Cost $c_i$", fontsize=20, weight="bold")
    plt.ylabel("Expected Improvement in Loss", fontsize=20, weight="bold")
    if len(experiments) > 1:
        plt.legend(loc="upper left", fontsize=15)
    plt.xlim([100 * cost_deviations(h)[0], 100 * cost_deviations(h)[-1]])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.grid(alpha=0.25)


def plot_penalty(exp, h=201, span=500):
    cost, m_star, total, _, _ = exp.mechanism()
    gamma_sigma_l = exp.config["gamma_sigma_l"]
    m = np.linspace(m_star - span, m_star + span, h)
    plt.plot(m, free_riding_penalty(m, cost, m_star, total, gamma_sigma_l) + m * cost, color="tab:red")
    optimum = free_riding_penalty(m_star, cost, m_star, total, gamma_sigma_l) + m_star * cost
    plt.plot(m_star, optimum, "h", color="tab:blue", markersize=8, label="True Optimal Contribution")
    plt.xlabel("Data Contributed $m_i$", fontsize=20, weight="bold")
    plt.ylabel("Free-Riding Penalty + Data Costs", fontsize=20, weight="bold")
    plt.legend(loc="best", fontsize=15)
    plt.xticks(m[::50])
    plt.xlim([m[0], m[-1]])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.grid(alpha=0.25)


def figures(results_dir):
    """Yield (family, path relative to the figures folder, draw function taking an rng)."""
    def exp(dataset, label, template, runs=3):
        return Experiment(results_dir, dataset, label, template, runs)

    settings = {}
    for dataset in ("cifar10", "mnist"):
        prefix = "fact-random-sandwich-uniform-cost"
        settings[dataset] = {
            "iid": exp(dataset, dataset, prefix + "-run{}-" + dataset + "-16devices"),
            "noniid6": exp(dataset, dataset, prefix + "-noniid-0.6-run{}-" + dataset + "-16devices"),
            "noniid3": exp(dataset, dataset, prefix + "-noniid-0.3-run{}-" + dataset + "-16devices"),
        }
    settings["ham"] = {"iid": exp("ham10000", "ham", "fact-sandwich-uniform-cost-run{}-ham10000-10devices")}

    for group in settings.values():
        for setting, e in group.items():
            stem = f"{e.num_agents}agents-{e.label}"
            yield "acc", f"acc/{setting}-{stem}-acc.jpg", lambda rng, e=e: plot_curves(e, "acc-top1", rng)
            yield "loss", f"loss/{setting}-{stem}-loss.jpg", lambda rng, e=e: plot_curves(e, "loss", rng)
            yield "histogram", f"histogram/{setting}-loss-histogram-{stem}.jpg", lambda rng, e=e: plot_loss_histogram(e)
        iid, experiments = group["iid"], list(group.values())
        stem = f"{iid.num_agents}agents-{iid.label}"
        yield ("truthful", f"truthful/vary-dist-truthfulness-{stem}.jpg",
               lambda rng, g=experiments: plot_truthfulness(g, rng))
        yield "free-riding", f"free-riding/penalty-truthfulness-{stem}.jpg", lambda rng, e=iid: plot_penalty(e)

    for k in (1, 4, 8):
        e = exp("ham10000", "ham", f"fact-robustness-{k}-nonuniform-cost-iid-run{{}}-ham10000-10devices", runs=1)
        yield ("robustness", f"robustness/{k}-irrational-loss-histogram-10agents-ham.jpg",
               lambda rng, e=e: plot_loss_histogram(e))


FAMILIES = ["acc", "loss", "histogram", "truthful", "free-riding", "robustness"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="output")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--only", nargs="+", choices=FAMILIES, default=FAMILIES)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for family, path, draw in figures(args.results_dir):
        if family not in args.only:
            continue
        path = os.path.join(args.figures_dir, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        draw(np.random.default_rng(0))
        plt.savefig(path, dpi=200)
        plt.close()
        print(path)


if __name__ == "__main__":
    main()
