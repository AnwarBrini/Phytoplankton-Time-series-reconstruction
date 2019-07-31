import torch
from torchsummary import summary
from models.cnn_models import Cnn2, swish

if __name__ == "__main__":
    device = torch.device('cpu')
    net = Cnn2(swish)
    net = (net.float()).to(device)
    summary(net, input_size=(10, 178, 358),device='cpu')
