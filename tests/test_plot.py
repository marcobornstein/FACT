from pathlib import Path

import plot

ROOT = Path(__file__).resolve().parents[1]


def test_plot_regenerates_every_committed_figure(tmp_path):
    plot.main(["--results-dir", str(ROOT / "output"), "--figures-dir", str(tmp_path)])
    produced = {p.relative_to(tmp_path) for p in tmp_path.rglob("*.jpg")}
    committed = {p.relative_to(ROOT / "figures") for p in (ROOT / "figures").rglob("*.jpg")}
    assert produced == committed
