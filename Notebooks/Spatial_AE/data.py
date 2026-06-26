import os
import cv2
import csv
import torch
import numpy as np
import pandas as pd

from model import *

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
#####################################################################

def format_dataset(folder_path, env):
    data_path = folder_path + '/Datasets/' + env
    csv_filename = data_path + "/data.csv"

    img_filename_list = os.listdir(data_path)
    num_images = len(img_filename_list) -1 

    return data_path, csv_filename, num_images
#####################################################################

def Load_env_dataset(folder_path, env):
    # Load dataset every T-maze version
    data_path, csv_filename, num_images = format_dataset(folder_path, env)

    print("----------  ----------  NEW ENVIRONMENT ----------  ----------")
    print("Dataset Environment    =", env)
    print("Num. Images in Dataset =", num_images)
    print("Dataset Path =", data_path)
    print("CSV path     =", csv_filename)
    print()
    
    dataset = load_dataset(data_path)  # this array should have shape (n_samples, 3, 120, 160)
    print("Dataset Shape  =", dataset.shape)

    return dataset, csv_filename

#####################################################################

def update_data(csv_filename):
    data = pd.read_csv(csv_filename)
    #data = data[:num_images]
    
    position = [] 
    pos_step = []
    
    for i in range(len(data)):
        pos_step = [data.X[i], data.Y[i]]
        position.append(pos_step)
        
    position = np.array(position)
    print("Position Shape =", position.shape)

    return data, position

#####################################################################

def update_model_trial(control, experiment_n, env, trial, model_filename, foldername, folder_path, n_hidden):
    # Update model name for trial t
    if control == True:
        foldername = 'Control/Control-' + foldername
    else:
        foldername = 'Experimental/' +foldername
    
    model_folder_path = folder_path + '/Models/' + foldername
    model_path = model_folder_path + '/' + model_filename    

    model = Conv_AE(n_hidden=n_hidden)
    model.load_state_dict(torch.load(model_path))

    return model

#####################################################################

def check_plot_folders(plot_folder_path, plot_foldername):
    folders = [name for name in os.listdir(plot_folder_path) if os.path.isdir(os.path.join(plot_folder_path, name))]
    if (plot_foldername in folders) == False:
        print('Creating new plot folder =', plot_foldername)
        os.makedirs(plot_folder_path + '/' + plot_foldername)
        
    return folders
#####################################################################

def check_model_folders(model_path, model_foldername):
    folders = [name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))]
    if (model_foldername in folders) == False:
        print('Creating new model folder =', model_foldername)
        os.makedirs(model_path + '/' + model_foldername)
        
    return folders
#####################################################################

def load_csv(csv_filename, num_images):
    data = pd.read_csv(csv_filename)
    data = data[:num_images]
    position = [] 
    pos_step = []
    orientation = []
    
    for i in range(len(data)):
        pos_step = [data.X[i], data.Y[i]]
        position.append(pos_step)
        z = data.Z[i]
        orientation.append(z)
        
    position = np.array(position)
    orientation = np.array(orientation)
    
    return data, position, orientation
#####################################################################

def save_training_loss(loss_csv_filename, loss):
    with open(loss_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image_loss'])
    
        # If you want each element on a new line
        for item in loss:
            writer.writerow([item])
    print("Loss History saved at", loss_csv_filename)
#####################################################################

def load_loss_history(model_folder_path):
    loss_csv_filename = model_folder_path + '/loss_history.csv'
    
    history_loss = []
    with open(loss_csv_filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            value = row.get('Image_loss')
            history_loss.append(float(value))
    return history_loss

#####################################################################

def init_ratemap_cache(path, n_checkpoints, n_units=200, h=50, w=50, dtype=np.float32):
    """
    Create a new memmap .npy file initialized with NaNs.
    """
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=(n_checkpoints, n_units, h, w)
    )
    arr[:] = np.nan
    del arr  # flush

#####################################################################

def open_ratemap_cache(path, n_checkpoints, n_units=200, h=50, w=50, dtype=np.float32):
    """
    Open an existing cache (r+) or create it if missing.
    Returns a memmap array you can slice-assign into.
    """
    if not os.path.exists(path):
        init_ratemap_cache(path, n_checkpoints, n_units, h, w, dtype)
    arr = np.lib.format.open_memmap(path, mode="r+")
    
    # Optional: sanity check shape
    expected = (n_checkpoints, n_units, h, w)
    if tuple(arr.shape) != expected:
        raise ValueError(f"Cache shape mismatch at {path}. Found {arr.shape}, expected {expected}")
    return arr

#####################################################################

def update_experiment_ratemaps(exp_n, data_path, control=False):
    if control==True:
        CACHE_DIR = data_path + "/Models/Control/Control-Exp" + str(exp_n) + "_cache_ratemaps"
    else:
        CACHE_DIR = data_path + "/Models/Experimental/Exp" + str(exp_n) + "_cache_ratemaps"

    print('CACHE_DIR =' , CACHE_DIR)
    
    cache_path = os.path.join(CACHE_DIR, f"ratemaps.npy")
    
    arr = np.load(cache_path, mmap_mode="r")  # no full load into RAM
    return arr, CACHE_DIR