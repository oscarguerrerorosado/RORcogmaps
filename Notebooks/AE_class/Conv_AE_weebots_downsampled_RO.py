import os
import cv2
import numpy as np
import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, ConcatDataset


def testing_img(image):
    print(image.shape)
    print(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(image.shape)
    print(image)

def load_dataset(directory, file_format='.jpg'):
    '''
    Loads all .jpg images and the pose data per image from a given directory.

    Args:
        directory (str): path to the images to be loaded.
        file_format (str): format of the images. Accepted formats are .npy and .jpg.
        load_pose (bool): if True, it will also load the pose data.
        pose_filename (str): name of the file with the pose data. The accepted format is .npy.

    Returns:
        images (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) and normalized values [0,1].
        pose (2D numpy array): pose data with (x,y) coordinates and angle (in degrees; [0,360]), wit shape (n_samples, 3).
    '''
    images = []
    for i, filename in enumerate(sorted(os.listdir(directory))):
        if filename.endswith(file_format):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    images = np.array(images)

    if np.max(images) > 1:   # normalize to [0,1] if values are RGB [0,255].
        images = images/255.

    return images


class Conv_AE(nn.Module):
    def __init__(self, n_hidden=500):
        #print('negative_slope = 0.001')
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (320,240,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
        '''
        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 15, 20

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)
        

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)  
        x = F.relu(self.fc1(x))
        
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true,C_factor, alpha=0):
        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        # Whitening loss (batch whitening).
        hidden_constraint_loss = 0
        batch_size, hidden_dim = hidden.shape

        # SSCP matrix
        M = torch.mm(hidden.t(), hidden)

        # Covariance matrix
        #hidden_centered = hidden - torch.mean(hidden, dim=0, keepdim=True)
        #M = torch.mm(hidden_centered.t(), hidden_centered) / (batch_size-1)
        
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor*I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
        loss = recon_loss + hidden_constraint_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()


    
def create_dataloader(dataset, batch_size=256, reshuffle_after_epoch=True):
    '''
    Creates a DataLoader for Pytorch to train the autoencoder with the image data converted to a tensor.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        batch_size (int; default=32): the size of the batch updates for the autoencoder training.

    Returns:
        DataLoader (Pytorch DataLoader): dataloader that is ready to be used for training an autoencoder.
    '''
    if dataset.shape[-1] <= 3:
        '''print("PREV TRANSFORMATION")
        print("dataset.shape-------->", dataset.shape)
        print("dataset.shape[-1]---->", dataset.shape[-1])'''
        dataset = np.transpose(dataset, (0,3,1,2))
        '''print("POST TRANSFORMATION")
        print("dataset.shape-------->", dataset.shape)
        print("dataset.shape[-1]---->", dataset.shape[-1])'''
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


def calculate_distances(positions, center):
    distances = np.linalg.norm(positions - center, axis=1)
    return distances

def compute_probabilities(distances, decay_factor=1.0):
    probabilities = np.exp(-decay_factor * distances)
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    return probabilities

def create_biased_dataloader(dataset, probabilities, batch_size=256):
    if dataset.shape[-1] <= 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=probabilities, num_samples=len(probabilities), replacement=True)

    # Convert images to a tensor dataset
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())

    # Create DataLoader with the sampler
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, sampler=sampler)

    return dataloader

def create_mixed_biased_dataloader(dataset, probabilities,
                                   bias_ratio=0.5,     # fraction of extra biased samples
                                   batch_size=256):
    """
    Creates a DataLoader where:
      - Every image appears at least once per epoch.
      - Images near the center appear more often overall.

    Parameters
    ----------
    dataset : np.ndarray
        Images, shape (N, H, W, C) or (N, C, H, W).
    bias_ratio : float
        Fraction of additional samples per epoch (0.0–1.0 typical).
        e.g. 0.5 means add 50% more biased samples.
    batch_size : int
        Batch size for DataLoader.
    """
    
    N = len(dataset)
    probs = torch.as_tensor(probabilities, dtype=torch.float32)
    
    # --- Stage 1: uniform coverage (each sample once)
    if dataset.ndim == 4 and dataset.shape[-1] in (1, 3):
        dataset = np.transpose(dataset, (0, 3, 1, 2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(),
                                   torch.from_numpy(dataset).float())
    
    # --- Stage 2: biased oversampling ---
    num_extra = int(bias_ratio * N)
    if num_extra > 0:
        sampler = WeightedRandomSampler(weights=probs, num_samples=num_extra, replacement=True)
        
        # Create dataset copy for extra samples (same images, same targets)
        biased_loader = DataLoader(tensor_dataset, batch_size=batch_size, sampler=sampler)
        extra_indices = list(iter(biased_loader.sampler))
        extra_subset = torch.utils.data.Subset(tensor_dataset, extra_indices)
        mixed_dataset = ConcatDataset([tensor_dataset, extra_subset])
    else:
        mixed_dataset = tensor_dataset
        extra_indices = []
    
    # --- Final DataLoader (uniform + biased extra samples) --
    dataloader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, extra_indices



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



def ratemaps(embeddings, position, n_bins=50, filter_width=2, occupancy_map=[], n_bins_padding=0, normalize=True):
    '''
    Creates smooth ratemaps from latent embeddings (activity) and spatial position through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        occupancy_map (2D numpy array; default=[]): 2D matrix reflecting the occupancy time across the space, with shape (n_bins+2*n_bins_padding, n_bins+2*n_bins_padding).
        n_bins_padding (int; default=0): the number of extra pixels with 0 value that are added to every side of the arena.

    Returns:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    occ_prob = occupancy_map/np.sum(occupancy_map)

    # Add activation values to each cell in the ratemap and adds Gaussian smoothing.
    n_latent = embeddings.shape[1]
    ratemaps = np.zeros((n_latent, int(n_bins+2*n_bins_padding), int(n_bins+2*n_bins_padding)))
    for i in range(n_latent):
        ratemap_ = np.zeros((n_bins, n_bins))
        for ii, c in enumerate(embeddings[:,i]):
            indx_x = pos_imgs_norm[ii,0]
            indx_y = pos_imgs_norm[ii,1]
            #ratemaps[i, indx_x, indx_y] += c
            ratemap_[indx_x, indx_y] += c
        ratemaps[i] = np.pad(ratemap_, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)
        if np.any(ratemaps[i]):
            ratemaps[i] = np.abs(ratemaps[i])
            #ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = gaussian_filter(ratemaps[i], filter_width) 
            #ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
            if len(occupancy_map) > 0:
                ratemaps[i] = ratemaps[i]/occ_prob
            if normalize == True:
                ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
        
    return ratemaps


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

def format_centroids(all_num_fields, centroids, sizes):
    centroids_per_field = []
    sizes_per_field = []
    centroid_index = 0
    
    for i in range(len(all_num_fields)):

        if all_num_fields[i] != 0:
            centroids_append = centroids[centroid_index:centroid_index+all_num_fields[i]].tolist()
            sizes_append = sizes[centroid_index:centroid_index+all_num_fields[i]].tolist()
            centroid_index += all_num_fields[i]
        else:
            centroids_append = [[0, 0]]
            sizes_append = [[0, 0]]


        centroids_per_field.append(centroids_append)
        sizes_per_field.append(sizes_append)
    
    return centroids_per_field, sizes_per_field

def plot_ratemaps(r, plot_path, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,20), dpi=600)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(r[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/100ratemaps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/100units.png', format='png')
    plt.show()


def plot_single_ratemap_density(r, unit, all_num_fields, sizes_per_field, centroids_per_field, plot_path, figsize=(3,3), save=False):
    print('Number of place fields = ' + str(all_num_fields[unit]))
    print('Size of place fields = ' + str(sizes_per_field[unit]))
    print('YX position of place fields = ' + str(centroids_per_field[unit]))

    fig = plt.figure(figsize=figsize)
    im = plt.imshow(r[unit], cmap='hot', origin='lower')

    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Activation', rotation=270, labelpad=12)
    
    if centroids_per_field[unit] != [[[0, 0]]]:
        for i in range(len(centroids_per_field[unit])):
            plt.scatter(centroids_per_field[unit][i][1], centroids_per_field[unit][i][0], color='green', marker='x', s=30)
    if save:
        fig.savefig(plot_path + '/Example_place_field.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_path + '/Example_place_field.png', format='png')
    plt.show()


def polarmaps(embeddings, angles, n_bins=20):
    '''
    Creates polarmaps from embedding activity and angle orientation through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        angles (list or 1D numpy array): list or 1D array containing the orientation angle (in radians or degrees) through 
                                         time, with shape (n_samples,).
        n_bins (int; default=20): resolution of the discretization of angles from which the polarmaps will be computed.

    Returns:
        polarmaps (2D numpy array): 2D matrix containing the polarmaps associated to all embedding units, with 
                                    shape (n_latent, n_bins).
    '''
    # Normalize orientation with respect to resolution to convert orientation to polarmap indices
    orien_imgs_norm = np.copy(angles)
    orien_imgs_norm = orien_imgs_norm/np.max(orien_imgs_norm)
    orien_imgs_norm = orien_imgs_norm*n_bins
    orien_imgs_norm = orien_imgs_norm.astype(int)
    
    # Add activation values to each cell in the ratemap and adds Gaussian smoothing
    n_latent = embeddings.shape[1]
    polarmaps = np.zeros((n_latent, n_bins))
    for i in range(n_latent):
        for ii, c in enumerate(embeddings[:,i]):
            indx = orien_imgs_norm[ii]
            polarmaps[i, indx-1] += c
        if np.any(polarmaps[i]):
            polarmaps[i] = polarmaps[i]/np.max(polarmaps[i])
        
    return polarmaps


def plot_polarmaps(p, plot_path, n_bins=20, n_cells_plot=30, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,16), dpi=600)
    
    for i in range(n_cells_plot):

        bottom = 0.4

        theta = np.linspace(0.0, 2*np.pi, n_bins, endpoint=False)
        radii = p[i]
        width = (2*np.pi) / (n_bins-1)

        ax = plt.subplot(5,6,i+1, polar=True)
        plt.title('Unit '+str(i+1))
        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        ax.set_theta_zero_location("W")

        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r / 5.))
            bar.set_alpha(0.8)

    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/Polar_maps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/Polar_maps.png', format='png')
    plt.show()

def stats_place_fields(ratemaps, peak_as_centroid=True, min_pix_cluster=0.02, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Runs a simple clustering algorithm to identify place fields, and compute their number, centroids, and sizes, for all ratemaps.

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        peak_as_centroid (bool; default=True): if True, the centroid will be taken as the peak of the place field; if False, it will take the 'center of mass'.
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        all_num_fields (1D numpy array): array with the number of place fields per embedding unit, with shape (n_latent,).
        all_centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        all_sizes (1D numpy array): array with the sizes of all place fields across embedding units, with shape (total_n_place_fields,).
    '''
    all_num_fields = []
    all_centroids = []
    all_sizes = []
    for r in ratemaps:

        ratemap = r.copy()
        
        ## Params.
        total_area = ratemap.shape[0]*ratemaps.shape[1]
        cluster_min = total_area*min_pix_cluster  #50
        cluster_max = total_area*max_pix_cluster #1250
        
        ## Clustering.
        ratemap[ratemap <  ratemap.max()*active_threshold] = 0
        ratemap[ratemap >= ratemap.max()*active_threshold] = 1

        visited_matrix  = np.zeros_like(ratemap)

        # First pass of clustering.
        clusterd_matrix = np.zeros_like(ratemap)
        current_cluster = 1

        # go through every bin in the ratemap.
        for yy in range(1,ratemap.shape[0]-1):
            for xx in range(1,ratemap.shape[1]-1):
                if ratemap[  yy, xx ] == 1:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                clusterd_matrix[ yy,xx ] = clusterd_matrix[ yy+ty, xx+tx ]

                    if clusterd_matrix[ yy, xx ] == 0:
                        clusterd_matrix[ yy, xx ] = current_cluster
                        current_cluster += 1
                        
        # Refine clustering: neighbour bins to same cluster number.
        for yy in range(1,clusterd_matrix.shape[0]-1):
            for xx in range(1,clusterd_matrix.shape[1]-1):
                if clusterd_matrix[  yy, xx ] != 0:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                if clusterd_matrix[ yy+ty, xx+tx ] != clusterd_matrix[  yy, xx ]:
                                    clusterd_matrix[ yy+ty, xx+tx ] = clusterd_matrix[  yy, xx ]
                  
        ## Quantify number of place fields.
        clusters_labels = np.delete(np.unique(clusterd_matrix), np.where(  np.unique(clusterd_matrix) == 0 ) )
        n_place_fields_counter = 0
        clusterd_matrix_ = np.copy(clusterd_matrix)
        clusters_labels_ = np.copy(clusters_labels)
        for k in range(clusters_labels.size):
            n_bins = np.where(clusterd_matrix == clusters_labels[k])[0].size
            if cluster_min <= n_bins <= cluster_max:
                n_place_fields_counter += 1
            else:
                clusterd_matrix_[np.where(clusterd_matrix_==clusters_labels[k])] = 0
                clusters_labels_ = np.delete(clusters_labels_, np.where(clusters_labels_ == clusters_labels[k]) )

        all_num_fields.append(n_place_fields_counter)
        
        ## Compute centroids.
        centroids = []
        for k in clusters_labels_:
            if peak_as_centroid:  # compute centroid as the peak of the place field.
                x, y = np.unravel_index(np.argmax( r * (clusterd_matrix_==k) ), r.shape)
                #x = np.argmax(  r * (clusterd_matrix_==k) ) 
                #y = np.argmax(  r * (clusterd_matrix_==k) )
            else:  # compute the centroid as weighted sum ('center of mass').
                w_x = r[np.where(clusterd_matrix_==k)[0], :].sum(axis=1)
                w_x = w_x/w_x.sum()
                x = np.sum(w_x * np.where(clusterd_matrix_==k)[0])
                
                w_y = r[:, np.where(clusterd_matrix_==k)[1]].sum(axis=0)
                w_y = w_y/w_y.sum()
                y = np.sum(w_y * np.where(clusterd_matrix_==k)[1])
            centroids.append([x,y])

        all_centroids += centroids
        
        ## Compute sizes of place fields.
        sizes = []
        for k in clusters_labels_:
            n_bins = np.where(clusterd_matrix_ == k)[0].size
            sizes.append(n_bins)

        all_sizes += sizes
    
    return np.array(all_num_fields), np.array(all_centroids), np.array(all_sizes)




def plot_place_field_hist(num_fields, plot_path, save=False):
    '''
    TO DO.
    '''
    place_field_counts = np.histogram(num_fields, bins=np.max(num_fields)+1, density=True)[0]
    plt.figure(figsize=(5,4))
    plt.bar(np.arange(np.max(num_fields)+1), place_field_counts, width=1, color='black', alpha=1, edgecolor='white')
    plt.xlabel('# place fields', fontsize=20)
    plt.ylabel('prob.', fontsize=20)
    plt.yticks(np.linspace(0,1,6), np.linspace(0,1,6).round(1), fontsize=18)
    plt.xticks(np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), fontsize=18)
    plt.ylim(0,1)
    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/prob_place_field_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/prob_place_field_histogram.png', format='png')
    plt.show()