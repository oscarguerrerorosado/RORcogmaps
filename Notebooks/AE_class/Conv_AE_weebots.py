import os
import cv2
import numpy as np
import seaborn as sb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




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

class Control_Conv_AE(nn.Module):
    def __init__(self, n_hidden):

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
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor*I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size*hidden_dim)
        
        loss = recon_loss + hidden_constraint_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()


class Motivational_Conv_AE(nn.Module):
    def __init__(self, n_hidden=500, n_external=4):
        super(Motivational_Conv_AE, self).__init__()
        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 15, 20

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2 + n_external, n_hidden)
        self.pred_ext_input = nn.Linear(n_hidden, n_external)


        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encoder(self, x, ext_input):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * self.dim1 * self.dim2)
        # Concatenate external input to the flattened output before feeding to the linear layer
        x = torch.cat((x, ext_input), dim=1)
        x = F.relu(self.fc1(x))
        pred_ext_input = F.relu(self.pred_ext_input(x))
        return x, pred_ext_input

    def decoder(self, x):
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, self.dim1, self.dim2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

    def forward(self, x, ext_input):
        h, pred_ext_input = self.encoder(x, ext_input)
        out = self.decoder(h)
        return out, h, pred_ext_input

    def backward(self, optimizer, criterion, x, y_true, C_factor, alpha=0, ext_input=None):
        optimizer.zero_grad()

        y_pred, hidden, ext_input_pred = self.forward(x, ext_input)

        recon_loss = criterion(y_pred, y_true)
        recon_loss_ext_input = criterion(ext_input_pred, ext_input)

        # Whitening loss (batch whitening).
        hidden_constraint_loss = 0
        batch_size, hidden_dim = hidden.shape

        # SSCP matrix
        M = torch.mm(hidden.t(), hidden)

        # Covariance matrix
        I = torch.eye(hidden_dim, device='cuda')
        C = C_factor * I - M    # C = I - M    
        hidden_constraint_loss = alpha * torch.norm(C) / (batch_size * hidden_dim)
            
        loss = recon_loss + hidden_constraint_loss + recon_loss_ext_input #*1000
        loss.backward()

        optimizer.step()

        return recon_loss.item(), recon_loss_ext_input.item()



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
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)



def train_autoencoder(model, train_loader, external_inputs, C_factor, dataset=[], num_epochs=1000, learning_rate=1e-4, alpha=2e3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model = model.to('cuda')
    external_inputs=torch.Tensor(external_inputs).to('cuda')

    history = []
    ext_history = []
    embeddings = []
    if len(dataset) > 0:
        embeddings = [get_latent_vectors(dataset=dataset, model=model)]
    for epoch in range(num_epochs):
        running_loss = 0.
        running_ext_loss = 0.
        with tqdm(total=len(train_loader)) as pbar:
            for i, data in enumerate(train_loader, 0):
                
                inputs, _ = data #image
                inputs = inputs.to('cuda')
                # Prepare external input for this batch
                ext_inputs_batch = external_inputs[i * train_loader.batch_size: (i + 1) * train_loader.batch_size]

                loss, ext_loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, C_factor=C_factor, ext_input=ext_inputs_batch)
                running_loss += loss
                running_ext_loss += ext_loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Ext_loss: {running_ext_loss / len(train_loader):.4f}")

        history.append(running_loss / len(train_loader))
        ext_history.append(running_ext_loss / len(train_loader))

        if len(dataset) > 0:
            embeddings.append(get_latent_vectors(dataset=dataset, model=model))

    embeddings = np.array(embeddings)

    return history, embeddings, ext_history



def allo_predict(image, ext_input, model):
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2, 0, 1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)

    ext_input = torch.Tensor(ext_input).float().unsqueeze(0).to(next(model.parameters()).device)
    output_img, _, output_ext_input = model(image, ext_input)
    output_img = output_img[0].detach().cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0))

    output_ext_input = output_ext_input[0].detach().cpu().numpy()
    return output_img, output_ext_input


def control_predict(image, model):
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image)[0].detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img


def allo_get_latent_vectors(dataset, ext_input, model, batch_size=256):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    external_inputs=torch.Tensor(ext_input).to('cuda')
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to('cuda')
    latent_vectors = []
    with torch.no_grad():
        # Divide external inputs into batches
        ext_input_batches = [external_inputs[i * batch_size: (i + 1) * batch_size] for i in range(len(data_loader))]

        for batch, ext_inputs_batch in zip(data_loader, ext_input_batches):
            inputs, _ = batch
            inputs = inputs.to('cuda')
            latent = model(inputs, ext_inputs_batch)[1]
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors


def control_get_latent_vectors(dataset, model, batch_size=256):
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



def ratemaps(embeddings, position, n_bins=50, filter_width=2, occupancy_map=[], n_bins_padding=0):
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
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = gaussian_filter(ratemaps[i], filter_width) 
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
            if len(occupancy_map) > 0:
                ratemaps[i] = ratemaps[i]/occ_prob
                ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
        
    return ratemaps



def plot_ratemaps(r, plot_path, plot_title, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,20), dpi=600)
    plt.suptitle(plot_title, fontsize=50)
    #plt.subplots_adjust(top=1.5)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(r[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/100ratemaps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/100units.png', format='png')
    plt.show()


def plot_motivational_ratemaps(r, plot_path, plot_title, motivation, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(10,10), dpi=600)
    plt.suptitle(plot_title, fontsize=20)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.title('Unit ' + str(i+1))
        plt.imshow(r[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/100ratemapsM' + str(motivation) + '.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/100ratemapsM' + str(motivation) + '.png', format='png')
    plt.show()


def plot_single_ratemap_density(r, unit, all_num_fields, sizes_per_field, centroids_per_field, plot_path, figsize=(3,3), save=False):
    print('Number of place fields = ' + str(all_num_fields[unit]))
    print('Size of place fields = ' + str(sizes_per_field[unit]))
    print('YX position of place fields = ' + str(centroids_per_field[unit]))

    fig = plt.figure(figsize=figsize)
    plt.imshow(r[unit], cmap='hot', origin='lower')
    plt.title('Unit ' + str(unit + 1) )
    if centroids_per_field[unit] != [[[0, 0]]]:
        for i in range(len(centroids_per_field[unit])):
            plt.scatter(centroids_per_field[unit][i][1], centroids_per_field[unit][i][0], color='green', marker='x', s=30)
    if save:
        fig.savefig(plot_path + '/Example_place_field.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_path + '/Example_place_field.png', format='png')
    plt.show()

def plot_motivational_single_ratemap(r1, r2, r3, r4, unit, plot_path, figsize=(6,6), save=False):

    fig = plt.figure(figsize=figsize)
    plt.suptitle('Motivational ratemps for unit ' + str(unit + 1), fontsize=15)
    plt.subplot(221)
    plt.imshow(r1[unit], cmap='hot', origin='lower')
    plt.title('Motivation 1')
    plt.subplot(222)
    plt.imshow(r2[unit], cmap='hot', origin='lower')
    plt.title('Motivation 2')
    plt.subplot(223)
    plt.imshow(r3[unit], cmap='hot', origin='lower')
    plt.title('Motivation 3')
    plt.subplot(224)
    plt.imshow(r4[unit], cmap='hot', origin='lower')
    plt.title('Motivation 4')
   
    if save:
        fig.savefig(plot_path + '/motivational_single_ratemaps.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_path + '/motivational_single_ratemaps.png', format='png')
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
    plt.title('Place field probability', fontsize=20)
    plt.yticks(np.linspace(0,1,6), np.linspace(0,1,6).round(1), fontsize=18)
    plt.xticks(np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), fontsize=18)
    plt.ylim(0,1)
    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/prob_place_field_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/prob_place_field_histogram.png', format='png')
    plt.show()


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
            centroids_append = []
            sizes_append = []


        centroids_per_field.append(centroids_append)
        sizes_per_field.append(sizes_append)
    
    return centroids_per_field, sizes_per_field


def plot_centroid_hist(centroids_per_field, plot_path, save=False):
    max_centroids = 0
    for i in range(len(centroids_per_field)):
        if len(centroids_per_field[i]) > max_centroids:
            max_centroids = len(centroids_per_field[i])

    centroid_distribution = []

    for i in range(max_centroids+1):
        count=0
        for a in range(len(centroids_per_field)):
            if len(centroids_per_field[a]) == i:
                count += 1
        centroid_distribution.append(count)

    total_values = sum(centroid_distribution)
    centroid_percentage = np.array(centroid_distribution) / total_values * 100
    indices = range(len(centroid_percentage))

    plt.figure(figsize=(8, 6))
    plt.bar(indices, centroid_percentage, color='black')
    plt.xlabel('Num. of centroids', size=12)
    plt.ylabel('Percentage of units', size=12)
    plt.title('Distribution of place fields per unit', size=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    x_labels = list(indices)
    plt.xticks(indices, x_labels)
    if save:
        plt.savefig(plot_path + '/centroids_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/centroids_histogram.png', format='png')
    plt.show()


def plot_place_field_sizes_hist(sizes_per_field, plot_path, bins=6, save=False):
    sizes = []
    for i in range(len(sizes_per_field)):
        for a in range(len(sizes_per_field[i])):
            sizes.append(sizes_per_field[i][a])

    plt.figure(figsize=(8, 6))
    plt.hist(sizes, weights=np.ones(len(sizes)) / len(sizes), bins=bins, color='black')

    plt.xlabel('Place field size', size=12)
    plt.ylabel('Percentage of units', size=12)
    plt.title('Distribution of place fields sizes', size=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    #X Ticks
    max_value = max(sizes)
    size_ticks = []
    for i in range(bins):
        size_ticks.append(round(((max_value/bins) *i)  + (max_value/(bins*2))))    
    plt.xticks(size_ticks)
    
    if save:
        plt.savefig(plot_path + '/PF_size_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/PF_size_histogram.png', format='png')
    plt.show()


def raster_plot(unit_activation, title=''):
    # Create a figure with desired figsize
    fig, ax0 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the data with pcolor and set the vmax parameter
    c = ax0.pcolor(unit_activation, cmap='hot', vmax=1.)

    # Add color bar with an arbitrary maximum value of 1.5
    cbar = plt.colorbar(c, ax=ax0)
    cbar.set_label('Norm. Unit Actv.')  # Set the label of the color bar

    # Set title
    ax0.set_title('Raster plot for ' + title, fontsize=15, fontweight="bold")
    ax0.set_ylabel('Sorted Units')
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Tight layout
    fig.tight_layout()

    # Show the plot
    plt.show()


def ratemap_filtered_Gaussian(ratemap, std=2):
    '''
    Adds Gaussians filters to a ratemap in order to make it more spatially smooth.

    Args:
        ratemap (2D numpy array): unfiltered ratemap with the activity counts across space.
        std (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units). 

    Returns:
        new_ratemap (2D numpy array): original ratemap filtered with Gaussian smoothing.
    '''
    new_ratemap = gaussian_filter(ratemap, std)   
    return new_ratemap


def generate_occupancy_map(position, n_bins=50, filter_width=0, n_bins_padding=0, norm=True):
    '''
    Computes the occupancy map based on the position through time.

    Args:
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        padding_n (int; default=0): the number of extra pixels that are added to every side of the arena.

    Returns:
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).
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

    map_occ = np.zeros((n_bins, n_bins))
    for p in pos_imgs_norm:
        ind_x, ind_y = p
        map_occ[ind_x, ind_y] += 1

    map_occ = np.pad(map_occ, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)

    map_occ = ratemap_filtered_Gaussian(map_occ, filter_width)

    if norm:
        map_occ = map_occ/np.sum(map_occ, axis=(0,1))

    occupancy_map = map_occ.T

    return occupancy_map

def spatial_information(ratemaps, occupancy_map):
    '''
    Spatial information score (SI) as computed in Skaggs et al. 1996. The SI is computed per rate (i.e., embedding unit).

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        SI (1D numpy array): array with SI scores, in bit/spike, with shape (n_latent,).
    '''
    ratemaps_ = ratemaps[np.any(ratemaps, axis=(1,2))]
    FR = ratemaps_/(np.mean(ratemaps_, axis=(1,2))[:,np.newaxis,np.newaxis])
    OT = occupancy_map/np.sum(occupancy_map)
    log_FR = np.log2(FR, out=np.zeros_like(FR, dtype='float64'), where=(FR!=0))
    SI = np.sum(FR*OT*log_FR, axis=(1,2))
    
    return SI