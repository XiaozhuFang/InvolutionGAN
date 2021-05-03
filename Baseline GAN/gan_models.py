import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

# G(z)
class Generator_MLP(nn.Module):
    # initializers
    def __init__(self, batch_size=64, image_size=64, z_dim=100, mlp_dim=64):
        super(Generator_MLP, self).__init__()
        self.fc1 = nn.Linear(z_dim, mlp_dim*4)
        self.fc2 = nn.Linear(self.fc1.out_features, mlp_dim*8)
        self.fc3 = nn.Linear(self.fc2.out_features, mlp_dim*16)
        self.fc4 = nn.Linear(self.fc3.out_features, image_size)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))

        return x, None, None

class Discriminator_MLP(nn.Module):
    # initializers
    def __init__(self, batch_size=64, image_size=64, mlp_dim=64):
        super(Discriminator_MLP, self).__init__()
        self.fc1 = nn.Linear(image_size, mlp_dim*16)
        self.fc2 = nn.Linear(self.fc1.out_features, mlp_dim*8)
        self.fc3 = nn.Linear(self.fc2.out_features, mlp_dim*4)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.sigmoid(self.fc4(x))

        return x, None, None