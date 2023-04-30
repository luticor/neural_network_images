import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
import os
import pickle
import datetime

# if "DISPLAY" not in os.environ:
#     matplotlib.use("Agg")  # Use non-interactive backend for scripts running without a display
# else:
#     matplotlib.use("notebook")  # Use interactive backend for Jupyter notebooks
import matplotlib.pyplot as plt

import model

class NeuralNetworkTrainer:
    def __init__(self, dataloader, net=model.Net(nn_shape=(2,4,50,1)), n_epochs=20, lr=10**(-2.3), lr_schedule='auto', plot_func=None):
        self.dataloader = dataloader
        self.net = net
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_exp = np.log10(lr)
        self.lr_schedule = lr_schedule
        self.plot_func = plot_func
        self.loss_values = []

    def train(self, plot=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr).to(device)
        for epoch in range(self.n_epochs):
            if self.lr_schedule=='auto' or self.lr_schedule is None:
                for g in optimizer.param_groups:
                    g['lr'] = 10**(self.lr_exp - 2.*(epoch/(self.n_epochs-1)))
            elif  self.lr_schedule=='fixed':
                pass
            print('Epoch', epoch, 'of', self.n_epochs)
            loop = tqdm(self.dataloader, leave=True)
            for x_batch, y_batch in loop:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # Forward pass
                outputs = self.net(x_batch)
                loss = criterion(outputs, y_batch.unsqueeze(dim=-1))
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store the loss value
                self.loss_values.append(loss.cpu().item())

            # Print the loss
            print(f"Loss = {self.loss_values[-1]}")
            if plot:
                # Show current network progress
                if epoch in [0,1,3,6,9,19,39]:
                    self.plot_output()

    def plot_output(self):
        # Call plot function if it is defined
        if self.plot_func is not None:
            self.plot_func(self.net)
        else:
            print("self.plot_func is undefined.")


    def plot_loss(self):
        epochs_values = np.array(range(len(self.loss_values))) / len(self.dataloader)
        plt.plot(epochs_values, self.loss_values)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid()
        plt.show()

    def save_model(self, file_path):
        self.net.cpu()
        torch.save(self.net.state_dict(), file_path)

    def save_trainer(self, file_path=None):
        if file_path is None:
            now = datetime.datetime.now()
            date_string = now.strftime("%m%d_%H%M")
            file_path = f"working/model_{date_string}.pickle"
        # Send the model and optimizer back to CPU if they are on GPU
        self.net.cpu()
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_trainer(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


'''
To use the modules use the following commands:

nn_trainer = NeuralNetworkTrainer(dataloader, net=model.Net, n_epochs=20, lr=10**(-2.3), plot_func=None)
nn_trainer.train()
nn_trainer.plot_output()
nn_trainer.plot_loss()
nn_trainer.save_trainer()
nn_trainer = NeuralNetworkTrainer.load_trainer(file_path={filepath})
'''
