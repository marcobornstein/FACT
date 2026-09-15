"""Per-agent result logs: one text file per series, named r<rank>-<series>.log."""

import os
import time
from collections import defaultdict

import numpy as np


class Recorder:
    def __init__(self, folder, rank):
        self.folder = folder
        self.rank = rank
        self.series = defaultdict(list)

    def append(self, series, value):
        self.series[series].append(value)

    def flush(self):
        for name, values in self.series.items():
            np.savetxt(self._path(f"{name}.log"), values, delimiter=",")

    def save_array(self, name, array):
        np.save(self._path(name), array)

    def _path(self, filename):
        return os.path.join(self.folder, f"r{self.rank}-{filename}")


def create_run_folder(comm, path, description):
    """Create the run folder on rank 0 and return its path on every rank.

    An existing folder is never overwritten: a timestamp is appended instead.
    """
    if comm.Get_rank() == 0:
        if os.path.exists(path):
            path += time.strftime("-%Y%m%d-%H%M%S")
        os.makedirs(path)
        with open(os.path.join(path, "ExpDescription"), "w") as f:
            f.write(f"{description!r}\n")
    return comm.bcast(path if comm.Get_rank() == 0 else None, root=0)
