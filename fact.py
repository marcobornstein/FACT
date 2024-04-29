import torch
import numpy as np
import torchvision.models as models
import argparse
from config import configs
from mpi4py import MPI
from train_test import local_training, federated_training, federated_training_nonuniform
from utils.communicator import Communicator
from utils.loader import load_cifar10, load_mnist
from utils.recorder import Recorder
from utils.models import MNIST
from utils.truthfulness import agent_contribution, agent_gain_inflation, agent_gain_truth


if __name__ == '__main__':

    # parse dataset from command line
    parser = argparse.ArgumentParser(description='FACT Dataset Parser')
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()

    # determine config
    dataset = args.dataset
    config = configs[dataset]

    # determine hyper-parameters
    num_train_data = config['num_train_data']
    train_batch_size = config['train_bs']
    test_batch_size = config['test_bs']
    learning_rate = config['lr']
    epochs = config['epochs']
    log_frequency = config['log_frequency']
    marginal_cost = config['marginal_cost']
    local_steps = config['local_steps']
    uniform_cost = config['uniform_cost']
    non_iid = config['non_iid']
    alpha = config['dirichlet_value']
    seed = config['random_seed']
    name = config['name']

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # set seed for reproducibility
    torch.manual_seed(seed+rank)
    np.random.seed(seed+rank)

    # determine torch device available (default to GPU if available)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        dev = ["cuda:" + str(i) for i in range(num_gpus)]
        device = dev[gpu_id]

    else:
        num_gpus = 0
        device = "cpu"

    # initialize federated communicator
    FLC = Communicator(rank, size, comm, device)

    # initialize recorder
    recorder = Recorder(rank, size, config, name, dataset)

    # keep note of true and reported marginal costs
    used_cost = marginal_cost if uniform_cost else marginal_cost * 0.9  # np.random.normal(marginal_cost, marginal_cost / 10)
    recorder.save_costs(marginal_cost, used_cost)
    # marginal_cost = 1 / int(num_train_data / size) ** 2
    # print(marginal_cost)

    # compute amount of data to use
    num_data, data_cost = agent_contribution(used_cost, offset=1)
    print(used_cost)
    print('rank: %d, local optimal data: %d, reported marginal cost %.3E' % (rank, num_data, used_cost))

    # in order to partition data without overlap, share the amount of data each device will use
    all_data = np.empty(size, dtype=np.int32)
    comm.Allgather(np.array([num_data], dtype=np.int32), all_data)
    self_weight = num_data / np.sum(all_data)
    FLC.self_weight = self_weight

    # load CIFAR10 data
    if dataset == 'cifar10':
        trainloader, testloader = load_cifar10(all_data, rank, size, train_batch_size, test_batch_size, non_iid, alpha)
        model = models.resnet18()
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    elif dataset == 'mnist':
        trainloader, testloader = load_mnist(all_data, rank, size, train_batch_size, test_batch_size, non_iid, alpha)
        model = MNIST()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = None
    else:
        print('ERROR: Dataset Provided Is Not Valid.')
        exit()

    # use ResNet18
    criterion = torch.nn.CrossEntropyLoss()

    # synchronize models (so they are identical initially)
    FLC.sync_models(model)

    # save initial model for federated training
    model_path = recorder.saveFolderName + '-model.pth'
    if rank == 0:
        torch.save(model, model_path)

    # load model onto GPU (if available)
    model.to(device)

    # run local training (no federated mechanism)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Local Training...')

    loss_local = local_training(model, trainloader, testloader, device, criterion, optimizer, epochs, log_frequency,
                                recorder, scheduler)

    MPI.COMM_WORLD.Barrier()

    # reset model to the initial model
    model = torch.load(model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Federated Training...')

    loss_fed = federated_training(model, FLC, trainloader, testloader, device, criterion, optimizer, epochs,
                                  log_frequency, recorder, scheduler, local_steps=local_steps)

    MPI.COMM_WORLD.Barrier()

    # simulate the truthfulness mechanism
    agent_net_loss = loss_local - loss_fed
    net_losses = np.empty(size, dtype=np.int32)
    comm.Allgather(np.array([agent_net_loss], dtype=np.int32), net_losses)
    average_other_agent_loss = (np.sum(net_losses) - agent_net_loss) / (size - 1)

    if rank == 0:

        print(agent_net_loss)
        print(average_other_agent_loss)

        agent_net_benefit = agent_gain_truth(marginal_cost, used_cost, num_data, agent_net_loss,
                                             average_other_agent_loss, agents=1000, random=False)

        print(agent_net_benefit)
