import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle



def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    epoch_loss, all_y, all_y_pred = 0, [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, X_lengths, y) in enumerate(train_loader):
        # distribute data to device
        X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X), X_lengths)   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)  # mini-batch loss
        epoch_loss += F.cross_entropy(output, y, reduction='sum').item()  # sum up mini-batch loss

        y_pred = torch.max(output, 1)[1]  # y_pred != output

        # collect all y and y_pred in all mini-batches
        all_y.extend(y)
        all_y_pred.extend(y_pred)

        # to compute accuracy
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    epoch_loss /= len(train_loader)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    epoch_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    return epoch_loss, epoch_score


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y, all_y_pred = [], []
    with torch.no_grad():
        for X, X_lengths, y in test_loader:
            # distribute data to device
            X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X), X_lengths)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up minibatch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    check_mkdir(save_model_path)
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score
