import os
import numpy as np
import pandas as pd
import warnings
import sys
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
import skimage
from skimage.io import imread
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset


def plot_geoloc(train_coordinates, val_coordinates=None):
    """
    Args:
        train_coordinates: list of (lat,lon)
        val_coordinates: list of (lat,lon)
    Returns:
        MapboxPlot of region
    """

    if val_coordinates is not None:
        df = pd.DataFrame(train_coordinates+val_coordinates, columns=['lat', 'lon'])
        df.insert(2, "focus", [0]*len(train_coordinates)+[1]*len(val_coordinates), True)
    else:
        df = pd.DataFrame(train_coordinates, columns=['lat', 'lon'])
    mapbox_access_token = os.environ['mapbox_access_token'] # Mapbox access token
    px.set_mapbox_access_token(mapbox_access_token)
    if val_coordinates is not None:
        fig = px.scatter_mapbox(df, lon="lon", lat="lat", color='focus', zoom=1, color_continuous_scale=px.colors.diverging.Picnic)
    else:
        fig = px.scatter_mapbox(df, lon="lon", lat="lat", zoom=1, color_continuous_scale=px.colors.diverging.Picnic)
    return fig


def getImageSetDirectories(data_dir='data/chips', labels_filename='data/labels.csv', band_list=['img_ndwi'], test_size=0.1, plot_coords=True,
                           plot_class_imbalance=True, seed=123):
    """ Return list of list of paths to filenames for training and validation (KFold)
    Args:
        data_dir: str, path to chips folder.
        band_list: list of str, from img_ndwi, img_02, img_03, img_04, img_08 and optionally bg_ndwi, bg_02, bg_03, bg_04, bg_08.
        test_size: float, proportion of locations for validation
        plot_coords: bool, plot coordinates with mapbox
        plot_class_imbalance: bool, plot target histograms (train, val labels)
    Returns:
        train_img_paths, val_img_paths: list of (list of list of str)
        fig: mapbox plot of coordinates if plot_coords is True. Otherwise, returns None.
    """
    
    df_labels = pd.read_csv(labels_filename, dtype={'count': float})
    df_labels = df_labels[df_labels["count"] >= 0.0] # keep positive counts
    df_labels_groupby = df_labels.groupby("lat_lon")
    coordinates = np.array(list(df_labels_groupby.groups.keys()))
    train_coordinates, val_coordinates = train_test_split(coordinates, test_size=test_size, random_state=seed, shuffle=True) # split train/val coordinates
    print("Found {0} coordinates:{1} train / {2} val".format(len(coordinates), len(train_coordinates), len(val_coordinates)))
    
    def get_img_paths(coordinates):
        img_paths = []
        for subdir in coordinates:
            timestamps = df_labels_groupby.get_group(name = subdir)["timestamp"] # if count is negative, will not appear in the group
            for timestamp in timestamps:
                img_timestamp = []
                for band in band_list: # img_08, bg_ndwi, img_clp
                    if band.startswith('img_'):
                        img_timestamp.extend(glob.glob(os.path.join(data_dir, subdir, band + "*t_" + timestamp + "*.png"))) 
                    else:
                        img_timestamp.extend(glob.glob(os.path.join(data_dir, subdir, band +  "*.png")))
                if len(img_timestamp)==len(band_list): ##### sanity check (BUG for certain coords / chips)
                    img_paths.append(img_timestamp)
                else:
                    print('Assertion error', len(img_timestamp),len(band_list),img_timestamp)
        return np.array(img_paths)
    
    train_img_paths = get_img_paths(train_coordinates) # get list of filenames
    val_img_paths = get_img_paths(val_coordinates)
    
    fig = None
    if plot_coords is True:
        train_coords = [coord.replace('lat_','').split('_lon_') for coord in train_coordinates]
        train_coords = [ (float(coord[0].replace('_','.')), float(coord[1].replace('_','.'))) for coord in train_coords]
        val_coords = [coord.replace('lat_','').split('_lon_') for coord in val_coordinates]
        val_coords = [ (float(coord[0].replace('_','.')), float(coord[1].replace('_','.'))) for coord in val_coords]
        fig = plot_geoloc(list(train_coords), list(val_coords))
    
    if plot_class_imbalance is True:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
        plt.figure(1, figsize=(20,5))
        plt.subplot(121)
        df_labels[df_labels['lat_lon'].isin(train_coordinates)]['count'].hist(color='blue')
        plt.xlabel('label')
        plt.ylabel('counts (train)')
        plt.subplot(122)
        df_labels[df_labels['lat_lon'].isin(val_coordinates)]['count'].hist(color='red')
        plt.xlabel('label')
        plt.ylabel('counts (val)')
        plt.show()
        
    return train_img_paths, val_img_paths, fig


class S2_Dataset(Dataset):
    """ Derived Dataset class for loading imagery from an imset_dir."""

    def __init__(self, imset_dir, augment=True, crop_size=2, labels_filename='data/labels.csv'):
        super().__init__()
        self.img_paths = imset_dir
        self.augment = augment
        self.crop_size = crop_size # if self.augment is True
        self.df_labels = pd.read_csv(labels_filename)
                    
    def __len__(self):
        return len(self.img_paths)        

    def __getitem__(self, index):
        """ Returns an ImageSet for the given index (int)."""   
        
        if not isinstance(index, int):
            raise KeyError('index must be int')
            
        imset = {}
        imset['img'] = np.stack([imread(filename) for filename in self.img_paths[index]],0)
        
        filename = self.img_paths[index][0] # ex: /home/jovyan/data/chips/lat_43_09_lon_5_93/img_08_t_2020-02-17.png
        imset['filename'] = filename
        
        lat_lon = filename.split('/')[-2]
        timestamp = filename.split('/')[-1].replace('.png','')[-10:]
        index = (self.df_labels['lat_lon']==lat_lon) * (self.df_labels['timestamp']==timestamp)
        imset['y'] = float(self.df_labels[index]['count'].values)
        
        if self.augment is True:
            h_flip, v_flip = np.random.rand(1)>0.5, np.random.rand(1)>0.5 # random flip
            if v_flip:
                imset['img'] = imset['img'][:,::-1]
            if h_flip:
                imset['img'] = imset['img'][:,:,::-1]
            k = np.random.randint(4) # random rotate
            imset['img'] = np.rot90(imset['img'], k=k, axes=(1,2))
            crop_x = np.random.randint(0, self.crop_size)
            crop_y = np.random.randint(0, self.crop_size)
            imset['img'] = imset['img'][:, crop_x:-self.crop_size+crop_x, crop_y:-self.crop_size+crop_y]

        imset['img'] = torch.from_numpy(skimage.img_as_float(imset['img']))
        imset['y'] = torch.from_numpy(np.array([imset['y']]))
        return imset

    
def plot_dataset(dataset, n_frames=14, n_rows=2):
    """ Plot dataset images. """
    fig = plt.figure(figsize=(16,5))
    for t in range(n_frames):
        plt.subplot(n_rows,n_frames//n_rows,1+t)
        imset = dataset[t]
        x = imset['img']
        y = int(imset['y'])
        plt.imshow(-x[0], cmap='RdYlBu') # NIR band
        plt.xticks([])
        plt.yticks([])
        plt.title('Label {}'.format(y))
    fig.tight_layout()
    plt.show()

