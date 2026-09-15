"""Keep README.md in sync with the code."""

import re
import shlex
from pathlib import Path

import pytest

import plot
import train
from fact.config import DATASETS
from fact.mechanism import optimal_contribution

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def commands(script):
    for block in re.findall(r"```bash\n(.*?)```", README, flags=re.DOTALL):
        for line in block.splitlines():
            words = shlex.split(line.split("#")[0])
            if script in words:
                yield words[words.index(script) + 1:]


def test_readme_commands_parse():
    train_commands, plot_commands = list(commands("scripts/train.py")), list(commands("scripts/plot.py"))
    assert train_commands and plot_commands
    for argv in train_commands:
        train.parse_args(argv)
    for argv in plot_commands:
        plot.parse_args(argv)


def test_readme_hyperparameters_match_config():
    rows = {line.split("|")[1].strip(" `"): [cell.strip() for cell in line.split("|")[2:-1]]
            for line in README.splitlines() if re.match(r"\| `(cifar10|mnist|ham10000)` \|", line)}
    assert set(rows) == {"cifar10", "mnist", "ham10000"}
    for dataset, (model, optimizer, lr, batch, epochs, steps, cost, m_star, agents) in rows.items():
        config = DATASETS[dataset]
        assert model.split()[0].lower() == config["model"]
        assert optimizer.split()[0].lower() == config["optimizer"]
        assert float(lr) == config["lr"]
        assert int(batch) == config["batch_size"]
        assert int(epochs) == config["epochs"]
        assert int(steps) == config["local_steps"]
        assert float(cost) == pytest.approx(config["marginal_cost"], rel=1e-3)
        assert int(m_star) == optimal_contribution(config["marginal_cost"], config["gamma_sigma_l"])
        assert int(agents) == config["num_agents"]
