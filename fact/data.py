"""Datasets and the split of training data among agents."""

import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Channel statistics of HAM10000.
HAM_MEAN = [0.7630392, 0.5456477, 0.57004845]
HAM_STD = [0.1409286, 0.15261266, 0.16997074]

# Rejected class draws after which the Dirichlet split samples the remaining classes directly.
MAX_REJECTED_DRAWS = 10_000_000


def load_datasets(name, data_dir):
    """Return (train set, test set, train labels as an array)."""
    if name == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    elif name == "ham10000":
        # Created by scripts/prepare_ham10000.py.
        root = os.path.join(data_dir, "HAM10000")
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            transforms.ToTensor(), transforms.Normalize(HAM_MEAN, HAM_STD),
        ])
        test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(HAM_MEAN, HAM_STD)])
        train = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        test = datasets.ImageFolder(os.path.join(root, "test"), transform=test_tf)
    elif name == "fake":
        tf = transforms.ToTensor()
        train = datasets.FakeData(256, (1, 28, 28), 10, transform=tf)
        test = datasets.FakeData(64, (1, 28, 28), 10, transform=tf, random_offset=256)
        return train, test, np.array([int(label) for _, label in train])
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return train, test, np.asarray(train.targets)


def partition(targets, sizes, seed, dirichlet_alpha=None):
    """Split sample indices into disjoint shards, shard i holding sizes[i] samples.

    The split depends only on `seed`, so every agent computes the same one. Samples beyond
    sum(sizes) are left unused.
    """
    sizes = np.asarray(sizes, dtype=int)
    if sizes.sum() > len(targets):
        raise ValueError(f"agents need {sizes.sum()} samples but the training set has {len(targets)}")
    rng = np.random.RandomState(seed)
    if dirichlet_alpha is None:
        return np.split(rng.permutation(len(targets)), np.cumsum(sizes))[:-1]
    return _dirichlet_partition(np.asarray(targets), sizes, dirichlet_alpha, rng)


def _dirichlet_partition(targets, sizes, alpha, rng):
    """Label-skewed split: agent i draws its class proportions from Dir(alpha), as in FedDC (Gao et al., 2022)."""
    num_classes = targets.max() + 1
    class_indices = [np.flatnonzero(targets == c) for c in range(num_classes)]
    class_remaining = np.array([len(indices) for indices in class_indices])
    priors = np.cumsum(rng.dirichlet([alpha] * num_classes, size=len(sizes)), axis=1)

    remaining = sizes.copy()
    shards = [np.empty(n, dtype=np.int32) for n in sizes]
    while remaining.sum() != 0:
        agent = rng.randint(len(sizes))
        if remaining[agent] <= 0:
            continue
        remaining[agent] -= 1
        label = _draw_available_class(rng, priors[agent], class_remaining > 0)
        class_remaining[label] -= 1
        shards[agent][remaining[agent]] = class_indices[label][class_remaining[label]]
    return shards


def _draw_available_class(rng, cumulative_prior, available):
    """Draw a class from the prior, redrawing while the class has no samples left.

    Consumes exactly the uniforms of drawing one at a time, but vectorises long streaks of rejections,
    which occur once the classes an agent favours are exhausted.
    """
    label = np.argmax(rng.uniform() <= cumulative_prior)
    batch, drawn = 64, 1
    while not available[label]:
        if drawn >= MAX_REJECTED_DRAWS:
            indices = np.flatnonzero(available)
            mass = np.diff(cumulative_prior, prepend=0)[indices]
            return rng.choice(indices, p=mass / mass.sum() if mass.sum() > 0 else None)
        state = rng.get_state()
        labels = np.argmax(rng.uniform(size=(batch, 1)) <= cumulative_prior, axis=1)
        hits = np.flatnonzero(available[labels])
        if hits.size:
            rng.set_state(state)
            rng.uniform(size=hits[0] + 1)
            label = labels[hits[0]]
        drawn += batch
        batch = min(2 * batch, 1 << 20)
    return label


def class_weights(targets):
    """Loss weights 1 - (class frequency), used for the imbalanced HAM10000 labels."""
    counts = np.bincount(targets).astype(np.float32)
    return 1 - counts / counts.sum()


def agent_loaders(train, test, shard, batch_size, test_batch_size, num_workers=0):
    train_loader = DataLoader(Subset(train, shard), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
