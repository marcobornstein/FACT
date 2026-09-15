"""End-to-end checks under mpirun, at several numbers of agents (including one)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not found")


def mpirun(n, *args):
    launcher = ["mpirun", "-n", str(n)]
    if "Open MPI" in subprocess.run(["mpirun", "--version"], capture_output=True, text=True, check=False).stdout:
        launcher.insert(1, "--oversubscribe")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([*launcher, sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=600, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


@pytest.mark.parametrize("n", [1, 3, 4])
def test_fedavg(n):
    mpirun(n, "tests/mpi_check_fedavg.py")


@pytest.mark.parametrize("n, flags", [
    (1, []),
    (2, []),
    (3, ["--free-riders", "1"]),  # uneven step counts during federated training
    (3, ["--non-iid", "0.3", "--nonuniform-cost"]),
])
def test_train(tmp_path, n, flags):
    epochs = 2
    mpirun(n, "scripts/train.py", "--dataset", "fake", "--device", "cpu", "--epochs", str(epochs), "--rounds", "1000",
           "--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path), *flags)

    folder = tmp_path / "FAKE" / f"fact-fake-{n}devices"
    fed_losses = [np.loadtxt(folder / f"r{i}-fed-epoch-loss.log") for i in range(n)]
    for i in range(n):
        # All agents evaluate the same averaged model.
        np.testing.assert_allclose(fed_losses[i], fed_losses[0], rtol=1e-5)
        assert np.loadtxt(folder / f"r{i}-local-epoch-loss.log").shape == (epochs,)
        benefits = np.loadtxt(folder / f"r{i}-benefits.log")
        assert benefits.shape == (3,) and np.isfinite(benefits).all()
        assert np.load(folder / f"r{i}-expected-epsilon-benefit.npy").shape == (81,)
    steps = {len(np.loadtxt(folder / f"r{i}-fed-train-loss.log")) for i in range(n)}
    assert len(steps) == 1, "agents took different numbers of federated steps"
