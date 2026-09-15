"""Train agents locally and federatedly, then evaluate the FACT mechanism. One MPI process per agent.

    mpirun -n 16 python train.py --dataset cifar10
    mpirun -n 16 python train.py --dataset mnist --non-iid 0.3 --seed 2
    mpirun -n 10 python train.py --dataset ham10000 --nonuniform-cost --free-riders 4
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
from mpi4py import MPI

from fact.comm import FedAvg
from fact.config import DATASETS
from fact.data import agent_loaders, class_weights, load_datasets, partition
from fact.mechanism import expected_truthfulness_benefit, fact_loss, optimal_contribution
from fact.models import build_model
from fact.recorder import Recorder, create_run_folder
from fact.training import train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--name", default="fact",
                        help="run name; results go to <output-dir>/<group>/<name>-<dataset>-<agents>devices")
    parser.add_argument("--seed", type=int, default=1, help="agent i seeds its generators with seed + i")
    parser.add_argument("--non-iid", type=float, metavar="ALPHA",
                        help="Dirichlet label skew across agents (the paper uses 0.6 and 0.3); iid if omitted")
    parser.add_argument("--nonuniform-cost", action="store_true",
                        help="draw each agent's marginal cost from N(c, (c/10)^2) instead of using c")
    parser.add_argument("--free-riders", type=int, default=0, metavar="K",
                        help="the last K agents report their optimal contribution but train on a fraction of it")
    parser.add_argument("--free-rider-fraction", type=float, default=0.25)
    parser.add_argument("--fed-optimizer", choices=["sgd", "adam"],
                        help="optimizer for federated training (default: the dataset's optimizer)")
    parser.add_argument("--epochs", type=int, help="override the dataset's number of epochs")
    parser.add_argument("--batch-size", type=int, help="override the dataset's batch size")
    parser.add_argument("--lr", type=float, help="override the dataset's learning rate")
    parser.add_argument("--rounds", type=int, default=100_000, help="rounds of the simulated sandwich game")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args(argv)


def resolve_config(args):
    """Dataset defaults with command-line overrides applied."""
    config = {**vars(args), **DATASETS[args.dataset]}
    for key in ("epochs", "batch_size", "lr"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config["fed_optimizer"] = args.fed_optimizer or config["optimizer"]
    return config


def marginal_cost(config, nonuniform, rng):
    """The agent's true marginal cost: the dataset's cost, or a draw from N(c, (c/10)^2)."""
    cost = config["marginal_cost"]
    return rng.normal(cost, 0.1 * cost) if nonuniform else cost


def pick_device(choice, rank):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return f"cuda:{rank % torch.cuda.device_count()}" if choice == "cuda" else choice


def make_optimizer(name, model, config):
    if name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = None
    if name == "sgd" and config["lr_milestones"]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_milestones"], gamma=0.1)
    return optimizer, scheduler


def main(argv=None):
    args = parse_args(argv)
    config = resolve_config(args)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if not 0 <= args.free_riders <= size:
        raise ValueError(f"--free-riders must be between 0 and the number of agents ({size})")

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = pick_device(args.device, rank)

    # Each agent's marginal cost determines its locally optimal contribution (Theorem 1).
    cost = marginal_cost(config, args.nonuniform_cost, np.random.RandomState(seed))
    contributions = np.array(comm.allgather(optimal_contribution(cost, config["gamma_sigma_l"])))
    # The server aggregates with the contributions agents report, so free riders keep their full weight.
    weight = contributions[rank] / contributions.sum()
    sizes = contributions.copy()
    if args.free_riders:
        sizes[-args.free_riders:] = (args.free_rider_fraction * sizes[-args.free_riders:]).astype(int)

    run_folder = os.path.join(args.output_dir, config["output_group"], f"{args.name}-{args.dataset}-{size}devices")
    recorder = Recorder(create_run_folder(comm, run_folder, config), rank)
    recorder.append("marginal-costs", cost)
    print(f"[rank {rank}] marginal cost {cost:.3e}, optimal contribution {contributions[rank]}, "
          f"training on {sizes[rank]} samples, device {device}", flush=True)

    # Rank 0 loads (and downloads) the data first so that agents do not race on the same files.
    if rank != 0:
        comm.Barrier()
    train_set, test_set, targets = load_datasets(args.dataset, args.data_dir)
    if rank == 0:
        comm.Barrier()
    shard = partition(targets, sizes, args.seed, args.non_iid)[rank]
    train_loader, test_loader = agent_loaders(train_set, test_set, shard, config["batch_size"],
                                              config["test_batch_size"], config.get("num_workers", 0))

    weights = torch.tensor(class_weights(targets)).to(device) if config.get("class_weighted_loss") else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    model = build_model(config["model"], num_classes=int(targets.max()) + 1)
    fedavg = FedAvg(comm, weight)
    fedavg.broadcast(model)
    initial_state = copy.deepcopy(model.state_dict())
    model.to(device)

    comm.Barrier()
    if rank == 0:
        print("Local training", flush=True)
    optimizer, scheduler = make_optimizer(config["optimizer"], model, config)
    local_loss = train(model, train_loader, test_loader, loss_fn, optimizer, device, config["epochs"],
                       len(train_loader), recorder, "local", scheduler, log_frequency=config["log_frequency"])

    comm.Barrier()
    if rank == 0:
        print("Federated training", flush=True)
    model.load_state_dict(initial_state)
    optimizer, scheduler = make_optimizer(config["fed_optimizer"], model, config)
    steps_per_epoch = max(comm.allgather(len(train_loader)))
    fed_loss = train(model, train_loader, test_loader, loss_fn, optimizer, device, config["epochs"],
                     steps_per_epoch, recorder, "fed", scheduler, fedavg=fedavg, local_steps=config["local_steps"],
                     log_frequency=config["log_frequency"])

    # FACT: each agent's reward is the average improvement of the other agents.
    net_improvements = np.array(comm.allgather(local_loss - fed_loss))
    others = (net_improvements.sum() - net_improvements[rank]) / (size - 1) if size > 1 else 0.0
    agent_fact_loss = fact_loss(local_loss, others, size)
    benefit = expected_truthfulness_benefit(cost, others, size, rounds=args.rounds, rng=np.random.default_rng(seed))
    for value in (net_improvements[rank], others, agent_fact_loss):
        recorder.append("benefits", value)
    recorder.flush()
    recorder.save_array("expected-epsilon-benefit", benefit)
    print(f"[rank {rank}] local loss {local_loss:.4f}, federated loss {fed_loss:.4f}, "
          f"FACT loss {agent_fact_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
