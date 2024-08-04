import torch
import pandas as pd
import numpy as np
import glob
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from mpi4py import MPI
from train_test import local_training, federated_training
from utils.communicator import Communicator
from utils.recorder import Recorder
from config import configs
from utils.truthfulness import agent_contribution, truthfulness_mechanism


def set_parameter_requires_grad(m, feature_extracting):
    if feature_extracting:
        for param in m.parameters():
            param.requires_grad = False


def create_directory_structure(dataset, output_folder):
    """
    Creates a directory structure for PyTorch ImageFolder from a folder of images.

    Args:
    image_folder: Path to the folder containing images.
    output_folder: Path to the output folder.
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Assuming image names contain class labels (adjust accordingly)
    for idx, sample in dataset.iterrows():
        image_path = sample['path']
        class_name = str(sample['cell_type_idx'])  # Extract class from filename
        class_dir = os.path.join(output_folder, class_name)

        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        shutil.copy(image_path, class_dir)


if __name__ == '__main__':

    config = configs["ham10000"]
    batch_size = config['train_bs']
    random_seed = config['random_seed']
    learning_rate = config['lr']
    data_dir = config['data_path']
    log_frequency = config['log_frequency']
    num_epochs = config['epochs']
    name = config['name']
    marginal_cost = config['marginal_cost']
    local_steps = config['local_steps']

    # reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        dev = ["cuda:" + str(i) for i in range(num_gpus)]
        device = dev[gpu_id]

    else:
        num_gpus = 0
        device = "cpu"

    # initialize federated communicator & recorder
    FLC = Communicator(rank, size, comm, device)
    recorder = Recorder(rank, size, config, name, "ham10000")

    # keep note of true and reported marginal costs
    recorder.save_costs(marginal_cost)

    # compute amount of data to use
    num_data, data_cost = agent_contribution(marginal_cost, offset=1)
    print('rank: %d, local optimal data: %d, reported marginal cost %.3E' % (rank, num_data, marginal_cost))

    # in order to partition data without overlap, share the amount of data each device will use
    all_data = np.empty(size, dtype=np.int32)
    comm.Allgather(np.array([num_data], dtype=np.int32), all_data)
    self_weight = num_data / np.sum(all_data)
    FLC.self_weight = self_weight

    all_image_path = glob.glob(os.path.join(data_dir, 'HAM_10000_Images/', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # load data, link labels to images
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # determine mean/std of figures
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]

    # determine ground truth
    y = df_original['cell_type_idx']
    df_train, df_test = train_test_split(df_original, test_size=0.2, random_state=random_seed, stratify=y)
    class_frequency = df_train['cell_type_idx'].value_counts().sort_index()
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    num_classes = len(class_frequency.index)
    num_train_data = df_train.shape[0]

    # ensure no leftover data before split
    if np.sum(all_data) < num_train_data:
        remainder = num_train_data - np.sum(all_data)
        added_data, leftover = divmod(remainder, size)
        all_data += added_data
        all_data[0] += leftover

    # create_directory_structure(df_train, os.path.join(data_dir, 'HAM_Loader_Train'))
    # create_directory_structure(df_test, os.path.join(data_dir, 'HAM_Loader_Test'))

    # add weighting for class imbalance
    weights = class_frequency.to_numpy(dtype=np.float32)
    weights = 1 - weights / np.sum(weights)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    # loss_fn = torch.nn.CrossEntropyLoss()

    # get pretrained model
    model = models.resnet50(weights="IMAGENET1K_V2")
    set_parameter_requires_grad(model, False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    input_size = 224

    # synchronize models (so they are identical initially)
    FLC.sync_models(model)

    # save initial model for federated training
    model_path = recorder.saveFolderName + '-model.pth'
    if rank == 0:
        torch.save(model, model_path)

    # load model onto GPU (if available)
    model.to(device)

    # define the transformation of the train images.
    train_transform = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(20),
         transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
         transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std)])

    # define the transformation of the test images.
    test_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])

    # Define the training set using the table train_df and using the defined transitions (train_transform)
    training_set = datasets.ImageFolder(os.path.join(data_dir, 'HAM_Loader_Train'), transform=train_transform)

    # Create the splits
    splits = random_split(training_set, all_data)

    # Select the desired split
    selected_dataset = splits[rank]

    # load training data
    train_loader = DataLoader(selected_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Same for the test set:
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'HAM_Loader_Test'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # run local training (no federated mechanism)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Local Training...')

    loss_local = local_training(model, train_loader, test_loader, device, loss_fn, optimizer, epochs=num_epochs,
                                log_frequency=log_frequency, recorder=recorder, scheduler=None)

    MPI.COMM_WORLD.Barrier()

    # reset model to the initial model
    model = torch.load(model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print('Beginning Federated Training...')

    loss_fed = federated_training(model, FLC, train_loader, test_loader, device, loss_fn, optimizer, num_epochs,
                                  log_frequency, recorder, None, local_steps=local_steps)

    MPI.COMM_WORLD.Barrier()

    # simulate the truthfulness mechanism
    agent_net_loss = loss_local - loss_fed
    net_losses = np.empty(size, dtype=np.float64)
    comm.Allgather(np.array([agent_net_loss], dtype=np.float64), net_losses)
    average_other_agent_loss = (np.sum(net_losses) - agent_net_loss) / (size - 1)
    fact_loss, avg_benefit_random = truthfulness_mechanism(marginal_cost, num_data, agent_net_loss,
                                                           average_other_agent_loss, size, agents=1000,
                                                           rounds=100000, h=81, normal=True)
    recorder.save_benefits(agent_net_loss, average_other_agent_loss, fact_loss, avg_benefit_random)