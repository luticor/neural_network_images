import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self, nn_shape=(2, 4, 100, 1)):
        super(Net, self).__init__()
        self.num_inputs = nn_shape[0]
        self.num_layers = nn_shape[1]
        self.num_neurons = nn_shape[2]
        self.num_outputs = nn_shape[3]
        self.layers = nn.ModuleList() # create an empty list to store layers
        self.layers.append(nn.Linear(self.num_inputs, self.num_neurons)) # add the first layer
        
        for i in range(self.num_layers - 2): # add the remaining layers
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        
        self.layers.append(nn.Linear(self.num_neurons, self.num_outputs)) # add the output layer
    
    def forward(self, x):
        # for layer in self.layers[:-1]:
        #     x = F.relu(layer(x))
        for idx in range(len(self.layers)-1):
            x = F.relu(self.layers[idx](x))
        x = self.layers[-1](x)
        return x


class Net_With_Sigmoid(Net):
    def __init__(self, nn_shape=(2, 4, 100, 1)):
        super().__init__(nn_shape)

    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)
        return (x)
    

# Define the neural network architecture
class ResNet(nn.Module):
    def __init__(self, nn_shape=(2, 4, 100, 1), resnet_skip=2):
        super(ResNet, self).__init__()
        self.num_inputs = nn_shape[0]
        self.num_layers = nn_shape[1]
        self.num_neurons = nn_shape[2]
        self.num_outputs = nn_shape[3]
        self.resnet_skip = resnet_skip
        self.layers = nn.ModuleList() # create an empty list to store layers
        self.layers.append(nn.Linear(self.num_inputs, self.num_neurons)) # add the first layer
        
        for i in range(self.num_layers - 2): # add the remaining layers
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        
        self.layers.append(nn.Linear(self.num_neurons, self.num_outputs)) # add the output layer

    def forward(self, x):
        skip_connection = F.relu(self.layers[0](x))
        for idx in range(len(self.layers)-1):
            x = F.relu(self.layers[idx](x))
            if idx % self.resnet_skip == 1: #Add skip connection to every other layer
                x = x + skip_connection
                skip_connection = x
        x = self.layers[-1](x)
        return x
    

class ResNet_With_Sigmoid(ResNet):
    def __init__(self, nn_shape=(2, 4, 100, 1), resnet_skip=2):
        super().__init__(nn_shape), resnet_skip=resnet_skip

    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)
        return (x)