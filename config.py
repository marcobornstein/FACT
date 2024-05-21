configs = {
    'cifar10': {
        'num_train_data': 50000,
        'train_bs': 128,
        'test_bs': 512,
        'lr': 0.05,
        'marginal_cost': 1.024e-07,
        'local_steps': 6,
        'random_seed': 1,
        'test_frequency': 500,
        'log_frequency': 60,
        'test_batches': 30,
        'epochs': 100,
        'file_path': 'output',
        'non_iid': False,
        'dirichlet_value': 0.3,
        'uniform_cost': True,
        'name': 'fact-random-sandwich-uniform-cost-iid-run1'
    },

    'mnist': {
            'num_train_data': 60000,
            'train_bs': 128,
            'test_bs': 1024,
            'lr': 5e-4,
            'marginal_cost': 7.11111111111e-08,
            'local_steps': 6,
            'random_seed': 1,
            'log_frequency': 30,
            'test_batches': 30,
            'epochs': 100,
            'file_path': 'output',
            'non_iid': False,
            'dirichlet_value': 0.3,
            'uniform_cost': True,
            'name': 'fact-random-sandwich-uniform-cost-iid-run1'
        }
}
