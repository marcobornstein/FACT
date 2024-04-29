configs = {
    'cifar10': {
        'num_train_data': 50000,
        'train_bs': 128,
        'test_bs': 512,
        'lr': 0.05,
        'marginal_cost': 4e-5,
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
        'name': 'fact'
    },

    'mnist': {
            'num_train_data': 60000,
            'train_bs': 128,
            'test_bs': 1024,
            'lr': 1e-3,
            'marginal_cost': 1e-8,
            'local_steps': 6,
            'random_seed': 1,
            'log_frequency': 30,
            'test_batches': 30,
            'epochs': 50,
            'file_path': 'output',
            'non_iid': False,
            'dirichlet_value': 0.3,
            'uniform_cost': True,
            'name': 'fact'
        }
}
