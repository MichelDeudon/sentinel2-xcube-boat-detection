import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
import skimage
from skimage.io import imread

import torch
from torch.utils.data import Dataset


def plot_geoloc(train_coordinates, val_coordinates=None):
    """
    Args:
        train_coordinates: list of (lat,lon)
        val_coordinates: list of (lat,lon)
        mapbox_access_token: str, Mapbox access token
    Returns:
        MapboxPlot of region
    """

    if val_coordinates is not None:
        df = pd.DataFrame(train_coordinates+val_coordinates, columns=['lat', 'lon'])
        df.insert(2, "focus", [0]*len(train_coordinates)+[1]*len(val_coordinates), True)
    else:
        df = pd.DataFrame(train_coordinates, columns=['lat', 'lon'])
        
    mapbox_access_token = os.environ['mapbox_access_token']
    px.set_mapbox_access_token(mapbox_access_token)
    if val_coordinates is not None:
        fig = px.scatter_mapbox(df, lon="lon", lat="lat", color='focus', zoom=1, color_continuous_scale=px.colors.diverging.Picnic)
    else:
        fig = px.scatter_mapbox(df, lon="lon", lat="lat", zoom=1, color_continuous_scale=px.colors.diverging.Picnic)
    return fig


def getImageSetDirectories(data_dir='data/chips', band_list=['img_ndwi'], test_size=0.1, plot_coords=True):
    """ Return list of paths to filenames for training and validation.
    Args:
        data_dir: str, path to chips folder.
        band_list: list of str, from img_ndwi, img_02, img_03, img_04, img_08 and optionally bg_ndwi, bg_02, bg_03, bg_04, bg_08.
        test_size: float, proportion of locations for validation
        plot_coords: bool, plot coordinates with mapbox
    Returns:
        train_img_paths, val_img_paths: list of list of str
        fig: mapbox plot of coordinates if plot_coords is True. Otherwise, returns None.
    """
    
    coordinates = os.listdir(data_dir)
    train_coordinates, val_coordinates = train_test_split(coordinates, test_size=test_size, random_state=1, shuffle=True)
    
    if plot_coords:
        train_coords = [coord.replace('lat_','').split('_lon_') for coord in train_coordinates]
        train_coords = [ (float(coord[0].replace('_','.')), float(coord[1].replace('_','.'))) for coord in train_coords]
        val_coords = [coord.replace('lat_','').split('_lon_') for coord in val_coordinates]
        val_coords = [ (float(coord[0].replace('_','.')), float(coord[1].replace('_','.'))) for coord in val_coords]
        fig = plot_geoloc(list(train_coords), list(val_coords))
    else:
        fig = None

    def get_img_paths(coords):
        img_paths = []
        for subdir in coords:
            for filename in os.listdir(os.path.join(data_dir,subdir)):
                if filename.startswith(band_list[0]):
                    filenames = [os.path.join(data_dir,subdir,filename)]
                    for band in band_list[1:]:
                        if band.startswith('bg'):
                            filenames.append(os.path.join(data_dir,subdir,band+'.png'))
                        elif band.startswith('img'):
                            filenames.append(os.path.join(data_dir,subdir,filename.replace(band_list[0],band)))
                    img_paths.append(filenames)
        img_paths = np.array(img_paths)
        return img_paths
    
    train_img_paths = get_img_paths(train_coordinates)
    val_img_paths = get_img_paths(val_coordinates)
    return train_img_paths, val_img_paths, fig


class S2_Dataset(Dataset):
    """ Derived Dataset class for loading imagery from an imset_dir."""

    def __init__(self, imset_dir, augment=True):
        super().__init__()
        self.img_paths = imset_dir
        self.augment = augment
        self.n_loc = len(np.unique(['/'.join(filenames[0].split('/')[:-1]) for filenames in self.img_paths]))
                    
    def __len__(self):
        return len(self.img_paths)        

    def __getitem__(self, index):
        """ Returns an ImageSet for the given index (int)."""   
        
        if not isinstance(index, int):
            raise KeyError('index must be int')
            
        imset = {}
        imset['img'] = np.stack([imread(filename) for filename in self.img_paths[index]],0)
        imset['y'] = float(self.img_paths[index][0].split('.')[0].split('_y_')[-1])
        imset['filename'] = self.img_paths[index][0]
        imset['n'] = float(len([1 for file in os.listdir('/'.join(imset['filename'].split('/')[:-1])) if file.startswith('img_08')]))
        
        if self.augment:
            h_flip, v_flip = np.random.rand(1)>0.5, np.random.rand(1)>0.5 # random flip
            if v_flip:
                imset['img'] = imset['img'][:,::-1]
            if h_flip:
                imset['img'] = imset['img'][:,:,::-1]
            k = np.random.randint(4) # random rotate
            imset['img'] = np.rot90(imset['img'], k=k, axes=(1,2))

        imset['img'] = torch.from_numpy(skimage.img_as_float(imset['img']))
        imset['y'] = torch.from_numpy(np.array([imset['y']]))
        imset['n'] = torch.from_numpy(np.array([imset['n']]))
        return imset
    
    
def plot_class_imbalance(train_list, val_list):
    """ Plot Training/Validation class imbalance. """
    
    plt.figure(1, figsize=(16,5))
    plt.subplot(121)
    plt.hist([int('y_0' not in filename[0]) for filename in train_list], color='blue')
    plt.xlabel('label')
    plt.ylabel('counts')
    plt.title('Training class imbalance')
    plt.subplot(122)
    plt.hist([int('y_0' not in filename[0]) for filename in val_list], color='red')
    plt.xlabel('label')
    plt.ylabel('counts')
    plt.title('Validation class imbalance')
    plt.show()
    
    
def plot_dataset(dataset, n_frames=14, n_rows=2, cmap='gray'):
    """ Plot dataset images. """
    
    fig = plt.figure(figsize=(16,5))
    for t in range(n_frames):
        plt.subplot(n_rows,n_frames//n_rows,1+t)
        x = dataset[t]['img']
        y = int(dataset[t]['y'])
        plt.imshow(x[0], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title('Label {}'.format(y))
    fig.tight_layout()
    plt.show()
    