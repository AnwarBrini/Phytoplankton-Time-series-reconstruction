# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:48:04 2019

@author: anwar
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os

def get_default_device():
    """
    Use GPU if available else CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """
    Move Pytorch tensor to device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """
    Wrap dataloader to move data to device
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """
        yield a batch of data after moving it to device
        """
        for b in self.dl:
            yield to_device(b, self.device)   
    def __len__(self):
        """
        Number of batches
        """
        return len(self.dl)
    
class Interpolated_Img_Dataset(Dataset):
    
    def __init__(self, root_folder, input_file, output_file , transform=None, normalize=True):
        """
        Args:
            root_folder (String): path to input and output files
            input_file (String): npy file of the input data to be used by the NN
            output_file (String): npy file of the output data to be use by the NN
            transform (callable, Optional): Optional transform to be applied on
            a sample
        """
        self.root_folder = root_folder
        self.input_arr = np.load(os.path.join(self.root_folder, input_file ))
        self.output_arr = np.log10(np.load(os.path.join(self.root_folder, output_file)))
        self.transform = transform
        self.normalize = normalize
        self.mean_input = np.mean(self.input_arr, axis=(0,1,2))
        self.std_input = np.std(self.input_arr, axis=(0,1,2))
        self.mean_output = np.mean(self.output_arr)
        self.std_output = np.std(self.output_arr)        
        
    def __len__(self):
        return self.input_arr.shape[0]
    
    def __getitem__(self, idx):
        X = self.input_arr[idx,...]
        Y = self.output_arr[idx,...]
        
        if self.normalize:
            for ch in range(10):
                X[:,:,ch] = (X[:,:,ch] - self.mean_input[ch])/ self.std_input[ch] 
            Y = (Y - self.mean_output)/ self.std_output
            
        sample = {'X':X, 'Y':Y}
        
        if self.transform:
            for key in sample.keys():
                sample[key] = self.transform(sample[key])
        return sample
        
