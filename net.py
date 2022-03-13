import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,
                 input_size: int,
                 params: list,
                 output_size: int,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, params[0])])
        for i in range(len(params) - 1):
            self.layers.append(nn.Linear(params[i], params[i+1]))
        self.layers.append(nn.Linear(params[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = F.softmax(self.layers[-1](x), dim=0)
        return x.clone()