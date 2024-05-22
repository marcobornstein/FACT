# FACT

To run the code in serial: 
```
python fact.py --dataset cifar10
```

To run the code in parallel: 
```
mpirun -n 16 python fact.py --dataset cifar10
```

One can change the experiment configuration by altering the config.py file (*e.g.,* change to a non-iid setting).
Change the dataset to MNIST by replacing `cifar10` with `mnist` above.

One can reproduce plots by providing new path links in experiments.py and running:
```
python experiments.py
```

Packages Used:
- matplotlib (3.8.4)
- mpi4py (3.1.6)
- networkx (3.3)
- numpy (1.26.4)
- scipy (1.13.0)
- torch (2.3.0)
- torchvision (0.18.0)
- tqdm (4.66.4)
