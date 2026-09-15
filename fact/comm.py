"""Weighted FedAvg over MPI, one process per agent."""

import time


class FedAvg:
    """Averages model parameters across agents, weighting agent i by s_i = m_i / sum_j m_j (Alg. 1)."""

    def __init__(self, comm, weight):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.weight = weight

    def average(self, model):
        """Replace the local model with the weighted average of all agents' models.

        Returns the wall-clock time spent in the collective.
        """
        state = {k: v.detach().cpu() * self.weight for k, v in model.state_dict().items()}
        self.comm.Barrier()
        start = time.time()
        states = self.comm.allgather(state)
        self.comm.Barrier()
        elapsed = time.time() - start

        averaged = states[self.rank]
        for i, other in enumerate(states):
            if i != self.rank:
                for key in averaged:
                    averaged[key] += other[key]
        model.load_state_dict(averaged)
        return elapsed

    def broadcast(self, model, root=0):
        """Give every agent an identical copy of the root agent's model."""
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()} if self.rank == root else None
        model.load_state_dict(self.comm.bcast(state, root=root))
