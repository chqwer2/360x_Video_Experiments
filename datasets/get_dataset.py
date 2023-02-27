
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm










## ---------------------- Dataloaders ---------------------- ##
# for CRNN- varying lengths (frames)
class Dataset_CRNN_varlen(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, lists, labels, set_frame, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders, self.video_len = list(zip(*lists))
        self.set_frame = set_frame
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)


    def __getitem__(self, index):
        "Generates one sample of data"
        # select sample
        selected_folder = self.folders[index]
        video_len = self.video_len[index]
        select = np.arange(self.set_frame['begin'], self.set_frame['end'] + 1, self.set_frame['skip'])
        img_size = self.transform.__dict__['transforms'][0].__dict__['size']       # get image resize from Transformation
        channels = len(self.transform.__dict__['transforms'][2].__dict__['mean'])  # get number of channels from Transformation

        selected_frames = np.intersect1d(np.arange(1, video_len + 1), select) if self.set_frame['begin'] < video_len else []

        # Load video frames
        X_padded = torch.zeros((len(select), channels, img_size[0], img_size[1]))   # input size: (frames, channels, image size x, image size y)

        for i, f in enumerate(selected_frames):
            frame = Image.open(os.path.join(self.data_path, selected_folder, 'frame{:06d}.jpg'.format(f)))
            frame = self.transform(frame) if self.transform is not None else frame  # impose transformation if exists
            X_padded[i, :, :, :] = frame

        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor
        video_len = torch.LongTensor([video_len])

        return X_padded, video_len, y

## ---------------------- end of Dataloaders ---------------------- ##




## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##



