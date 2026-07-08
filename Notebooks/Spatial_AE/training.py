import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_autoencoder(model, train_loader, C_factor,  dataset=[], num_epochs=1000, learning_rate=1e-4, alpha=2e3):
    '''
    TO DO.
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model = model.to('cuda')

    history = []
    embeddings = []
    if len(dataset) > 0:
        embeddings = [ get_latent_vectors(dataset=dataset, model=model) ]
    for epoch in range(num_epochs):
        running_loss = 0.
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                inputs, _ = data
                inputs = inputs.to('cuda')

                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, C_factor=C_factor)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        history.append(running_loss/len(train_loader))

        if len(dataset) > 0:
            embeddings.append( get_latent_vectors(dataset=dataset, model=model) )

    embeddings = np.array(embeddings)

    return history, embeddings
#####################################################################


def predict(image, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().

    Args:
        image (3D numpy array): sample image with shape (n_channels, n_pixels_height, n_pixels_width).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.

    Returns:
        output_img (3D numpy array): output image with shape (n_pixels_height, n_pixels_width, n_channels)
    '''
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image)[0].detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img
#####################################################################


def get_latent_vectors(dataset, model, batch_size=256):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to('cuda')
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            latent = model(inputs.to('cuda'))[1]
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors
#####################################################################


def evaluate_reconstruction_loss(model, dataset, batch_size=16):
    '''
    Evaluates the reconstruction loss (MSE) of a trained autoencoder on a given dataset.

    Args:
        model (torch.nn.Module): Trained autoencoder model.
        dataset (numpy array): Dataset of images with shape (N, H, W, C) or (N, C, H, W).
        batch_size (int): Batch size for evaluation.

    Returns:
        float: Average reconstruction loss (MSE) over the dataset.
    '''

    model.eval()
    model.to('cuda')

    # Ensure correct shape: (N, C, H, W)
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0, 3, 1, 2))

    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')  # sum to get total loss
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x = x.to('cuda')
            y_pred, _ = model(x)
            total_loss += criterion(y_pred, x).item()
            total_samples += x.shape[0] * x.shape[2] * x.shape[3] * x.shape[1]  # total number of pixels

    avg_loss = total_loss / total_samples
    return avg_loss
#####################################################################

def img_reconstruction_loss(model, image, criterion=nn.MSELoss()):
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        if image.shape[-1] <= 4:
            image = np.transpose(image, (2, 0, 1))
        n_channels, h, w = image.shape
        image_tensor = torch.from_numpy(image).float().reshape(1, n_channels, h, w).to(next(model.parameters()).device)
        output = model(image_tensor)[0]  # take first element of tuple
        loss = criterion(output, image_tensor)
    return loss.item()
#####################################################################

def img_pixelwise_reconstruction_loss(model, image):
    model.eval()
    with torch.no_grad():
        if image.shape[-1] <= 4:
            image = np.transpose(image, (2, 0, 1))
        n_channels, h, w = image.shape
        image_tensor = torch.from_numpy(image).float().reshape(1, n_channels, h, w).to(next(model.parameters()).device)
        output = model(image_tensor)[0]
        error_map = ((output - image_tensor) ** 2).squeeze(0).mean(0)  # average over channels
    return error_map.cpu().numpy()
#####################################################################