"""Per-dataset hyperparameters (Section 6 and Appendix B of the paper).

`gamma_sigma_l` is the constant gamma * sigma^2 * L of the agent loss (Eq. 2). Together with
the marginal cost c it fixes each agent's locally optimal contribution m* = sqrt(gamma_sigma_l / 2c)
(Theorem 1): 3,125 samples for CIFAR-10, 3,750 for MNIST and 801 for HAM10000.
"""

DATASETS = {
    "cifar10": {
        "output_group": "CIFAR10",
        "num_agents": 16,
        "model": "resnet18",
        "optimizer": "sgd",
        "lr": 0.05,
        "lr_milestones": [50, 75],
        "batch_size": 128,
        "test_batch_size": 512,
        "epochs": 100,
        "local_steps": 6,
        "marginal_cost": 1.024e-07,
        "gamma_sigma_l": 2.0,
        "log_frequency": 60,
    },
    "mnist": {
        "output_group": "MNIST",
        "num_agents": 16,
        "model": "cnn",
        "optimizer": "adam",
        "lr": 5e-4,
        "lr_milestones": None,
        "batch_size": 128,
        "test_batch_size": 1024,
        "epochs": 100,
        "local_steps": 6,
        "marginal_cost": 7.11111111111e-08,
        "gamma_sigma_l": 2.0,
        "log_frequency": 30,
    },
    "ham10000": {
        "output_group": "HAM",
        "num_agents": 10,
        "model": "resnet50",
        "optimizer": "adam",
        "lr": 1e-3,
        "lr_milestones": None,
        "batch_size": 128,
        "test_batch_size": 128,
        "epochs": 50,
        "local_steps": 6,
        "marginal_cost": 7.79e-07,
        "gamma_sigma_l": 1.0,
        "log_frequency": 20,
        "class_weighted_loss": True,
        "num_workers": 4,
    },
    # Tiny random images for smoke tests; not used in the paper.
    "fake": {
        "output_group": "FAKE",
        "num_agents": 2,
        "model": "cnn",
        "optimizer": "adam",
        "lr": 1e-3,
        "lr_milestones": None,
        "batch_size": 16,
        "test_batch_size": 64,
        "epochs": 1,
        "local_steps": 2,
        "marginal_cost": 2.44140625e-04,  # 64 samples per agent
        "gamma_sigma_l": 2.0,
        "log_frequency": 2,
    },
}
