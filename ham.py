import torch
import pandas as pd
import numpy as np
import glob
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from PIL import Image
from train_test import local_training, federated_training
from utils.recorder import Recorder
from config import configs
from tqdm import tqdm


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


def train(train_dataloader, m, criterion, opt, epoch, total_loss_train, total_acc_train):
    m.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_dataloader)
    for i, data in enumerate(train_dataloader):

        images, labels = data

        N = images.size(0)
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        opt.zero_grad()
        outputs = m(images)

        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 10 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_dataloader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg, total_loss_train, total_acc_train


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    random_seed = 2024
    learning_rate = 1e-3
    batch_size = 64  # 32 original

    # reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    data_dir = 'data/HAM10000'
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

    # create_directory_structure(df_train, os.path.join(data_dir, 'HAM_Loader_Train'))
    # create_directory_structure(df_test, os.path.join(data_dir, 'HAM_Loader_Test'))

    # add weighting for class imbalance
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_frequency.to_numpy(dtype=np.float32)))
    # loss_fn = torch.nn.CrossEntropyLoss()

    # get pretrained model
    model_ft = models.resnet50(weights="IMAGENET1K_V2")
    set_parameter_requires_grad(model_ft, False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    input_size = 224

    # place model on device
    # device = torch.device('cuda:0') If GPU available
    device = torch.device('cpu')  # If using cpu
    model = model_ft.to(device)

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
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Same for the test set:
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'HAM_Loader_Test'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    config = configs["ham10000"]
    recorder = Recorder(0, 1, config, "test", "ham10000")

    print('beginning training...')

    loss_local = local_training(model, train_loader, test_loader, device, loss_fn, optimizer, epochs=10,
                                log_frequency=10, recorder=recorder, scheduler=None)

    """
    epoch_num = 10
    best_val_acc = 0
    total_loss_val, total_acc_val = [], []
    for e in tqdm(range(1, epoch_num + 1)):
        loss_train, acc_train, total_loss_val, total_acc_val = train(train_loader, model, loss_fn, optimizer, e,
                                                                     total_loss_val, total_acc_val)
    """
