"""scripts/reproduce_paper.sh must recreate the configurations recorded next to the committed results."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import train

ROOT = Path(__file__).resolve().parents[1]

# Fields taken from the runs' cost logs (marginal-costs.log) rather than their ExpDescription.
FROM_LOGS = {
    "fact-sandwich-uniform-cost-run2-ham10000-10devices": {"nonuniform_cost"},
    "fact-sandwich-uniform-cost-run3-ham10000-10devices": {"nonuniform_cost"},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run the script with a stub launcher and parse every command it issues."""
    tmp = tmp_path_factory.mktemp("stub")
    calls = tmp / "calls.jsonl"
    stub = tmp / "mpirun"
    stub.write_text(f"#!{sys.executable}\nimport json, sys\n"
                    f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n")
    stub.chmod(0o755)
    subprocess.run(["bash", "scripts/reproduce_paper.sh"], cwd=ROOT, check=True,
                   env={**os.environ, "MPIRUN": str(stub), "OUT": "stub-output"})

    parsed = []
    for line in calls.read_text().splitlines():
        call = json.loads(line)
        assert call[0] == "-n" and call[2:4] == ["python", "scripts/train.py"], call
        args = train.parse_args(call[4:])
        assert args.output_dir == "stub-output"
        config = train.resolve_config(args)
        folder = ROOT / "output" / config["output_group"] / f"{args.name}-{args.dataset}-{call[1]}devices"
        parsed.append((folder, int(call[1]), args, config))
    return parsed


def test_script_covers_every_committed_run_once(runs):
    committed = {p.parent for p in (ROOT / "output").glob("*/*/r0-benefits.log")}
    folders = [folder for folder, *_ in runs]
    assert len(folders) == len(set(folders))
    assert set(folders) == committed


def test_commands_match_recorded_configurations(runs):
    for folder, num_agents, _, config in runs:
        recorded = ast.literal_eval((folder / "ExpDescription").read_text())
        robustness = recorded.get("robustness") or {"use": False}
        expected = {
            "num_agents": num_agents,
            "batch_size": recorded["train_bs"],
            "lr": recorded["lr"],
            "epochs": recorded["epochs"],
            "seed": recorded["random_seed"],
            "marginal_cost": recorded["marginal_cost"],
            "non_iid": recorded["dirichlet_value"] if recorded["non_iid"] else None,
            "nonuniform_cost": not recorded["uniform_cost"],
            "free_riders": robustness["irrational_agents"] if robustness["use"] else 0,
        }
        if robustness["use"]:
            expected["free_rider_fraction"] = robustness["fr_factor"]
        for field in FROM_LOGS.get(folder.name, ()):
            del expected[field]
        assert {key: config[key] for key in expected} == expected, folder.name


def test_recorded_marginal_costs_follow_from_the_seeds(runs):
    for folder, num_agents, args, config in runs:
        for rank in range(num_agents):
            cost = train.marginal_cost(config, args.nonuniform_cost, np.random.RandomState(args.seed + rank))
            assert np.loadtxt(folder / f"r{rank}-marginal-costs.log") == pytest.approx(cost, rel=1e-12), folder.name
