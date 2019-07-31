#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:40:32 2019

@author: abrini
"""

import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import progressbar
from dataset import Interpolated_Img_Dataset
from models.cnn_models import Cnn1, swish

def reconstruct(data_loader, data_set, mask, model, output_folder):
    """
    Parameters
    ----------
    data_loader : torch Dataloader
    mask : torch tensor
    model : torch model
    output_folder : str
    Returns
    -------
    None.

    """
    with progressbar.ProgressBar(max_value=len(data_loader)) as recon_bar:
        for i, data in enumerate(data_loader, 0):
            #Getting input and output
            inputs, labels = data["X"], data["Y"]
            year = i//12 + 1998
            mon = i%12 + 1
            n_t, n_ch, n_lat, n_lon = (labels.float()).shape
            #forward pass
            outputs = model(inputs.float()).detach()            
            #Reshaping to apply mask
            outputs = outputs.reshape([n_t, n_ch, n_lat*n_lon])
            #Applying mask
            outputs[:, :, ~mask] = np.nan
            #Reshaping Back
            outputs = outputs.reshape([n_t, n_ch, n_lat, n_lon])
            #Rescaling
            #outputs = (np.power(10, outputs) * data_set.std_output) + data_set.mean_output
            #Saving Data
            np.save('{}/Chl_{}{:02d}'.format(output_folder, year, mon),
                   outputs)
            recon_bar.update(i)

def get_args():
    """
    Returns
    -------
    args : args parser
        arguments.

    """
    parser = argparse.ArgumentParser(description='Data reconstruction')
    parser.add_argument('--model-file', '-m', required=True, type=str,
                        help='Path for model to be saved')
    parser.add_argument('--data-path', '-i', required=True, type=str,
                        help='input data')
    parser.add_argument('--output-folder', '-o', required=True, type=str,
                        help='output folder for the reconstructed data')
    parser.add_argument('--mask-file', default="data/inter_mask.pt", type=str,
                        help='Masking file path default(data/inter_mask.pt)')
    args = parser.parse_args()
    return  args

def main():
    args = get_args()
    #Loading model and pretrained weights
    model = Cnn1(swish)
    #model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_file))
    #Loading data with batch=1 for reconstruction
    transform = transforms.Compose([transforms.ToTensor()])
    data_set = Interpolated_Img_Dataset(args.data_path, 'X.npy', 'Y.npy',
                                        transform=transform, normalize=True)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, num_workers=2)
    #Loading mask file
    mask = torch.load(args.mask_file)
    #Reconstruction
    reconstruct(data_loader, data_set, mask, model, args.output_folder)


if __name__ == "__main__":
    main()
