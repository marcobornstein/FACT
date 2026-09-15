"""FedAvg checks that need several MPI processes; launched by tests/test_mpi.py."""

import torch
from mpi4py import MPI

from fact.comm import FedAvg

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Weighted average: agent i holds parameters equal to i and has weight proportional to i + 1.
weights = [i + 1 for i in range(size)]
model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
with torch.no_grad():
    for parameter in model.parameters():
        parameter.fill_(float(rank))
FedAvg(comm, weights[rank] / sum(weights)).average(model)
expected = sum(i * w for i, w in enumerate(weights)) / sum(weights)
for parameter in model.parameters():
    torch.testing.assert_close(parameter.detach(), torch.full_like(parameter, expected))

# Broadcast: every agent ends up with the root's randomly initialised model.
torch.manual_seed(rank)
model = torch.nn.Linear(4, 4)
FedAvg(comm, 1.0).broadcast(model)
states = comm.allgather({k: v.clone() for k, v in model.state_dict().items()})
for state in states:
    for key, value in state.items():
        torch.testing.assert_close(value, states[0][key], rtol=0, atol=0)
