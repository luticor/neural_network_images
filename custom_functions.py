import numpy as np
import torch
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

def image_to_dataframe(img):
    '''
    This function takes in a PIL grayscale Image using:
        img = Image.open('{path_to_image}').convert('L')
    Then it converts the pixels into datapoints in a Pandas dataframe.

    Args:
        img (PIL.Image): Grayscale Image object from PIL

    Returns:
        df (pd.DataFrame): Dataframe of normalized pixel coordinates and normalized pixel values. 
            All values will be between 0 and 1. The vertical axis will be inverted with the bottom at 0 and top at 1.
    '''
    pixels = img.getdata()
    width, height = img.size
    #Extract the individual coordinates and pixel values as a list
    x_coords = []
    y_coords = []
    intensity = []
    for i in range(height):
        for j in range(width):
            x_coords.append(j)
            y_coords.append(height-i) #Flip y axis so bottom is 0 top is highest value
            intensity.append(pixels[i*width + j])
    # Assemble the Dataframe after turning the lists into arrays and normalizing by the maximum values
    df = pd.DataFrame({'x': np.array(x_coords)/width, 'y': np.array(y_coords)/height, 'z': np.array(intensity)/255.})
    return df

def df_to_dataloader(df, batch_size = 1024, shuffle=True, verbose=True):
    '''
    Converts dataframe of points to a Torch Dataloader.
    
    Args:
        df(pd.DataFrame): Dataframe of normalized pixel coordinates and normalized pixel values. All values will be between 0 and 1.
        batch_size (int): Batchsize for the dataloader and neural network training
        verbose (bool): Flag to allow print statements
    
    Returns:
        dataloader (torch.utils.data.Dataloader): A PyTorch dataloader that iterates through the data for training.
    '''
    #Convert DataFrame to tensors
    x_tensor = torch.tensor(df.iloc[:,0:2].values, dtype=torch.float)
    y_tensor = torch.tensor(df.iloc[:,-1].values, dtype=torch.float)

    # Create an instance of the dataset and dataloader
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if verbose:
        print('Batches per Epoch:',len(dataloader))
        print('x_tensor.shape', x_tensor.shape)
        print('y_tensor.shape', y_tensor.shape)

    return dataloader


def plot_performance(net, scale=8, batch_size=64, width = 1920, height = 1080):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Custom function to plot results of image learning.
    xlin = np.linspace(0, 1, width // scale)
    ylin = np.linspace(0, 1, height // scale)
    xv, yv = np.meshgrid(xlin, ylin)

    xv_tensor = torch.tensor(xv, dtype=torch.float)
    yv_tensor = torch.tensor(yv, dtype=torch.float)

    reduced_tensor = torch.stack((xv_tensor, yv_tensor), dim=-1)
    reduced_dataset = TensorDataset(reduced_tensor, torch.zeros(size=(len(reduced_tensor), 1)))
    reduced_dataloader = DataLoader(reduced_dataset, batch_size=batch_size, shuffle=False)
    net.to(device)
    with torch.no_grad():
        output_list = []
        for x_batch, _ in reduced_dataloader:
            x_batch = x_batch.to(device)
            outputs = net(x_batch)
            output_list.append(outputs.cpu())

        outputs = torch.cat(output_list, dim=0).numpy()
        
    plt.figure(figsize=(12, 8))
    plt.imshow(outputs, cmap='Greys_r', origin='lower')
    plt.show()

