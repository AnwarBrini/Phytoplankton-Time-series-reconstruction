# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:46:34 2019

@author: anwar
"""

from torch import nn
import torch

def swish(input_vector):
    """ 
    Parameters
    ----------
    x : Torch Tensor, Numpy array

    Returns
    -------
    TYPE Torch Tensor
        swish function of the input.

    """
    return input_vector * torch.sigmoid(input_vector)

class Cnn0(nn.Module):
    """
    CNN0 Class with Encode-Decode via ConvTranspose layers
    """
    def __init__(self):
        super(Cnn0, self).__init__()
        self.activation_layer = swish
        self.conv1_1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(32, 64, 2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 2, padding=1)
        self.dropout1 = nn.Dropout(0.85)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(64, 128, 2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 2, padding=1)
        self.dropout2 = nn.Dropout(0.85)
        self.pool3 = nn.MaxPool2d(2)
        self.trans4_1 = nn.ConvTranspose2d(128, 64, 10, dilation=(1, 2),
                                           stride=(2, 2), padding=1)
        self.trans5_1 = nn.ConvTranspose2d(64, 32, 20, dilation=(1, 3),
                                           stride=(2, 2), padding=1, output_padding=(0, 1))
        self.trans6_1 = nn.ConvTranspose2d(32, 1, 32, dilation=(2, 3),
                                           stride=(1, 1), padding=2)

    def forward(self, x):
        layer_out = x
        layer_out = self.conv1_1(layer_out)
        layer_out = self.activation_layer(layer_out)
        layer_out = self.activation_layer(self.conv1_2(layer_out))
        layer_out = self.pool1(layer_out)
        layer_out = self.activation_layer(self.conv2_1(layer_out))
        layer_out = self.dropout1(self.activation_layer(self.conv2_2(layer_out)))
        layer_out = self.pool2(layer_out)
        layer_out = self.activation_layer(self.conv3_1(layer_out))
        layer_out = self.dropout2(self.activation_layer(self.conv3_2(layer_out)))
        layer_out = self.pool3(layer_out)
        layer_out = self.activation_layer(self.trans4_1(layer_out))
        layer_out = self.activation_layer(self.trans5_1(layer_out))
        layer_out = self.trans6_1(layer_out)
        return layer_out

class ConvBlock(nn.Module):
    """
    Convolution block of 3 Convolution layers with conv layer (no batch normalization)
    """
    def __init__(self, activation_func, in_ch, out_ch, conv_depth=5, k_size=3, bnorm=True):
        super(ConvBlock, self).__init__()
        self.activation_func = activation_func
        self.use_bnorm = bnorm
        self.conv_depth = conv_depth
        assert isinstance(self.conv_depth, int) and self.conv_depth > 1, "conv_depth must be an integer > 1 "
        assert self.activation_func is not None, "activation_func can't be None"
        self.conv_1 = nn.Conv2d(in_ch, out_ch, k_size, padding=1)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, k_size, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        layer_out = x
        layer_out = self.activation_func(self.conv_1(layer_out))
        for _ in range(self.conv_depth - 1):
            layer_out = self.conv_2(layer_out)
            if self.use_bnorm:
                layer_out = self.norm(layer_out)
            layer_out = self.activation_func(layer_out)
        return layer_out

class Cnn1(nn.Module):
    """
    CNN1 Class with Encode-Decode via ConvTranspose layers
    """
    def __init__(self, activation_func, conv_depth=3, k_size=3):
        super(Cnn1, self).__init__()
        self.activation_func = activation_func
        self.conv_depth = conv_depth
        self.k_size = k_size
        # Maxpool layer
        self.down_layer = nn.MaxPool2d(2)
        # 4 Convolutional blocks for encoding
        self.e_conv_0 = ConvBlock(self.activation_func, 10, 16,
                                self.conv_depth, self.k_size, False)
        self.e_conv_1 = ConvBlock(self.activation_func, 16, 32,
                                self.conv_depth, self.k_size, False)
        self.e_conv_2 = ConvBlock(self.activation_func, 32, 64,
                                self.conv_depth, self.k_size)
        self.e_conv_3 = ConvBlock(self.activation_func, 64, 128,
                                self.conv_depth, self.k_size)
        # 4 Convolutional blocks for decoding and ConvTranspose to upscale
        self.up_0 = nn.ConvTranspose2d(128, 64, (12, 23),
                                       output_padding=(0, 1), dilation=(2, 2))
        self.d_conv_0 = ConvBlock(self.activation_func, 64, 64,
                                self.conv_depth, self.k_size)
        self.up_1 = nn.ConvTranspose2d(64, 32, (23, 46),
                                       output_padding=(1, 0), dilation=(2, 2))
        self.d_conv_1 = ConvBlock(self.activation_func, 32, 32,
                                self.conv_depth, self.k_size)
        self.up_2 = nn.ConvTranspose2d(32, 16, (23, 46), padding=(0, 1),
                                       output_padding=(1, 1), dilation=(4, 4))
        self.d_conv_2 = ConvBlock(self.activation_func, 16, 16,
                                self.conv_depth, self.k_size, False)
        self.out = nn.Conv2d(16, 1, self.k_size, padding=1)

    def forward(self, x):
        # Encoding and saving states
        x_0 = self.e_conv_0(x)
        x_1 = self.e_conv_1(self.down_layer(x_0))
        x_2 = self.e_conv_2(self.down_layer(x_1))
        layer_out = self.e_conv_3(self.down_layer(x_2))
        # Decoding and adding saved states from previous layers
        layer_out = self.up_0(layer_out)
        layer_out = layer_out.add(x_2)
        layer_out = self.d_conv_0(layer_out)
        layer_out = self.up_1(layer_out)
        layer_out = layer_out.add(x_1)
        layer_out = self.d_conv_1(layer_out)
        layer_out = self.up_2(layer_out)
        layer_out = layer_out.add(x_0)
        layer_out = self.d_conv_2(layer_out)
        layer_out = self.out(layer_out)
        return layer_out


class Cnn2(nn.Module):
    """
    CNN2 Class with Encode-Decode via ConvTranspose layers
    """
    def __init__(self, activation_func, conv_depth=3, k_size=3):
        super(Cnn2, self).__init__()
        self.activation_func = activation_func
        self.conv_depth = conv_depth
        self.k_size = k_size
        # Maxpool layer
        self.down_layer = nn.AvgPool2d(2)
        # 4 Convolutional blocks for encoding
        self.e_conv_0 = ConvBlock(self.activation_func, 10, 16,
                                self.conv_depth, self.k_size)
        self.e_conv_1 = ConvBlock(self.activation_func, 16, 32,
                                self.conv_depth, self.k_size)
        self.e_conv_2 = ConvBlock(self.activation_func, 32, 64,
                                self.conv_depth, self.k_size)
        # 4 Convolutional blocks for decoding and ConvTranspose to upscale
        # Tblock
        self.up_1_0 = nn.ConvTranspose2d(64, 32, (7, 12))
        self.up_conv1_0 = nn.Conv2d(32, 32, self.k_size, padding=1)
        self.up_1_1 = nn.ConvTranspose2d(32, 16, (16, 34))
        self.up_conv1_1 = nn.Conv2d(16, 32, self.k_size, padding=1)
        self.up_1_2 = nn.ConvTranspose2d(32, 32, (13, 24), dilation=(2, 2))

        self.d_conv_1 = ConvBlock(self.activation_func, 32, 32,
                                self.conv_depth, self.k_size)
        # Tblock
        self.up_2_0 = nn.ConvTranspose2d(32, 16, (7, 17), dilation=(2, 2))
        self.up_conv2_0 = nn.Conv2d(16, 16, self.k_size, padding=1)
        self.up_2_1 = nn.ConvTranspose2d(16, 8, (16, 35), dilation=(2, 2))
        self.up_conv2_1 = nn.Conv2d(8, 16, self.k_size, padding=1)
        self.up_2_2 = nn.ConvTranspose2d(16, 16, (24, 40), dilation=(2, 2),
                                         output_padding=1)

        self.d_conv_2 = ConvBlock(self.activation_func, 16, 16,
                                self.conv_depth, self.k_size)
        self.out = nn.Conv2d(16, 1, self.k_size, padding=1)

    def forward(self, x):
        # Encoding and saving states
        x_0 = self.e_conv_0(x)
        x_1 = self.e_conv_1(self.down_layer(x_0))
        layer_out = self.e_conv_2(self.down_layer(x_1))
        layer_out = self.up_1_2(self.up_conv1_1(
                self.up_1_1(self.up_conv1_0(self.up_1_0(layer_out)))))
        layer_out = layer_out.add(x_1)
        layer_out = self.d_conv_1(layer_out)
        layer_out = self.up_2_2(self.up_conv2_1(
                self.up_2_1(self.up_conv2_0(self.up_2_0(layer_out)))))
        layer_out = layer_out.add(x_0)
        layer_out = self.d_conv_2(layer_out)
        layer_out = self.out(layer_out)
        return layer_out
