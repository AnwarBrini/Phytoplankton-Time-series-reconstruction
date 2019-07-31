# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:04:39 2019

@author: anwar
"""
import argparse
import time
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchsummary import summary
import progressbar
from utils import EarlyStopping
from dataset import Interpolated_Img_Dataset
from models.cnn_models import Cnn1, swish

def get_train_loader(data_set, train_sampler, batch_size):
    """
    Parameters
    ----------
    data_set : torch dataset object
    train_sampler : subsetsampler
        train_slice sampler.
    batch_size : int

    Returns
    -------
    train_loader : torch Dataloader
        train_set Dataloader.

    """
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=2)
    return train_loader


def create_loss_optimizer(net, learning_rate):
    """
    Parameters
    ----------
    net : torch nn object
        neural network.
    learning_rate : float

    Returns
    -------
    loss : torch loss
    optimizer : torch optimizer

    """
    #Loss
    loss = torch.nn.MSELoss()
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return (loss, optimizer)


def train_network(net, device, data_set, train_sampler, val_sampler,
                  batch_size, n_epochs, save_path, mask_file,
                  patience=20, learning_rate=0.001):
    """
    Train nn
    Compute loss only on unmasked dataLogger

    Parameters
    ----------
    net : torch nn object
    device : torch device
    data_set : Dataset object
        DESCRIPTION.
    train_sampler : SubsetSampler
    val_sampler : SubsetSampler
    batch_size : int
    n_epochs : int
    save_path : string
        path of saved pt model.
    mask_file : string
        path to torch file of the mask.
    patience : int, optional
        patience before early stopping. The default is 20.
    learning_rate : float, optional
         The default is 0.001.

    Returns
    -------
    net : torch nn object
        trained neural net.

    """
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    #Setting the loggerDataParallel
    logger = SummaryWriter('./logs/{}'.format(time.time()))
    #Get training data
    train_loader = get_train_loader(data_set, train_sampler, batch_size)
    val_loader = torch.utils.data.DataLoader(data_set, batch_size=1,
                                             sampler=val_sampler, num_workers=2)
    n_batches = len(train_loader)
    #Moving nn to device
    net = (net.float()).to(device)
    #Create our loss and optimizer functions
    loss, optimizer = create_loss_optimizer(net, learning_rate)
    #Time for printing
    training_start_time = time.time()
    #Loading Mask file
    mask = torch.load(mask_file)
    #Model Summary
    summary(net, input_size=(10, 178, 358))
    #Initializing early stopping variable
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    try:
        #Loop for n_epochs
        with progressbar.ProgressBar(max_value=n_epochs) as epoch_bar:
            for epoch in range(n_epochs):
                running_loss = 0.0
                print_every = n_batches // 5
                #start_time = time.time()
                total_train_loss = 0
                with progressbar.ProgressBar(max_value=len(train_loader)) as train_bar:
                    for i, data in enumerate(train_loader, 0):
                        #Get inputs
                        inputs, labels = data["X"].to(device), data["Y"].to(device)
                        n_t, n_ch, n_lat, n_lon = (labels.float()).shape
                        #Set the parameter gradients to zerototal_train_loss
                        optimizer.zero_grad()
                        #Forward pass, Mask Output
                        outputs = net(inputs.float())
                        #Reshaping to apply mask
                        outputs = outputs.reshape([n_t, n_ch, n_lat*n_lon])
                        labels = (labels.float()).reshape([n_t, n_ch, n_lat*n_lon])
                        #Applying mask
                        outputs[:, :, ~mask] = labels[:, :, ~mask]
                        #Reshaping to Compute Loss
                        outputs = outputs.reshape([n_t, n_ch, n_lat, n_lon])
                        labels = labels.reshape([n_t, n_ch, n_lat, n_lon])
                        #Backward pass, optimize
                        loss_size = loss(outputs.float(), labels.float())
                        loss_size.backward()
                        optimizer.step()
                        #Print statistics
                        running_loss += loss_size.item()
                        total_train_loss += loss_size.item()
                        train_bar.update(i)
                        #Print every 10th batch of an epoch
                        if (i + 1) % (print_every + 1) == 0:
                            #Reset running loss and time
                            running_loss = 0.0
                            #start_time = time.time()
                print("\n Train loss = {:.4f}".format(total_train_loss / len(train_loader)))
                epoch_bar.update(epoch)
                #At the end of the epoch, do a pass on the validation set
                total_val_loss = 0
                with progressbar.ProgressBar(max_value=len(val_loader)) as val_bar:
                    for i, data in enumerate(val_loader, 0):
                        inputs, labels = data["X"].to(device), data["Y"].to(device)
                        n_t, n_ch, n_lat, n_lon = (labels.float()).shape
                        #Forward passTYPE
                        val_outputs = net(inputs.float())
                        #Reshaping to apply mask
                        val_outputs = val_outputs.reshape([n_t, n_ch, n_lat*n_lon])
                        labels = (labels.float()).reshape([n_t, n_ch, n_lat*n_lon])
                        #Applying mask
                        val_outputs[:, :, ~mask] = labels[:, :, ~mask]
                        #Rshaping to compute loss
                        val_outputs = val_outputs.reshape([n_t, n_ch, n_lat, n_lon])
                        labels = (labels.float()).reshape([n_t, n_ch, n_lat, n_lon])
                        #Loss
                        val_loss_size = loss(val_outputs.float(), labels.float())
                        total_val_loss += val_loss_size.item()
                        val_bar.update(i)
                print("Validation loss = {:.4f}".format(total_val_loss / len(val_loader)))
                #Loggin Train and Validation Losses
                info = {'loss-Train': total_train_loss / len(train_loader),
                        'loss-Val': total_val_loss / len(val_loader)}
                for tag, value in info.items():
                    logger.add_scalar(tag, value, epoch+1)
                #Early stop if patience is surpassed
                early_stopping((total_val_loss / len(val_loader)), net)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
        torch.save(net.cpu().state_dict(), save_path)
    except KeyboardInterrupt:
        print("Keyboard Interruption")
        print("Saving Model.......")
        torch.save(net.cpu().state_dict(), 'INTERRUPTED.pth')
        print("{} Saved".format(save_path))
    return net


def get_args():
    # TODO add the rest of the cnn_model arguments in order to apply bayesian optimization
    """
    Returns
    -------
    args : args parser
        arguments.

    """
    parser = argparse.ArgumentParser(description='Training Cnn Pytorch Model')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate default(0.001)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='random seed (default 42)')
    parser.add_argument('--batch-size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', '-e (default 500)', default=500, type=int,
                        help='total epochs to run')
    parser.add_argument('--patience', '-p', default=50, type=int,
                        help='patience before early stopping (default 50)')
    parser.add_argument('--split-frac', '-f', default=0.7, type=float,
                        help='train-split fraction (default 0.7)')
    parser.add_argument('--data-path', '-d', default='data/interpolated', type=str,
                        help='Path for folder containing data default (data/interpolated)')
    parser.add_argument('--mask-file', default="data/inter_mask.pt", type=str,
                        help='Masking file path default(data/inter_mask.pt)')
    parser.add_argument('--model-file', '-m', required=True, type=str,
                        help='Path for model to be saved')
    args = parser.parse_args()
    return  args


def main():
    """
    Main

    Returns
    -------

    """
    args = get_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Train Validation Split
    assert args.split_frac <= 1
    n_train = int(args.split_frac*216)
    n_val = 216 - n_train

    transform = transforms.Compose([transforms.ToTensor()])
    data_set = Interpolated_Img_Dataset(args.data_path, 'X.npy', 'Y.npy',
                                        transform=transform, normalize=False)

    #RandomSampler
    train_sampler = SubsetRandomSampler(np.arange(n_val, n_train + n_val, dtype=np.int64))
    val_sampler = SubsetRandomSampler(np.arange(n_val, dtype=np.int64))
    #Initializing model with desired activation function
    net = Cnn1(activation_func=swish)
    #net = torch.nn.DataParallel(net)
    #Resume Training from previously saved state_dict
    if args.resume:
        net.load_state_dict(torch.load(args.model_file))
    _ = train_network(net, device, data_set, train_sampler,
                                                val_sampler,
                                                args.batch_size,
                                                args.epoch,
                                                args.model_file,
                                                args.mask_file,
                                                args.patience, args.lr)


if __name__ == "__main__":
    main()
