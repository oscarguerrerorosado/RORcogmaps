import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, ConcatDataset

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
#####################################################################

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