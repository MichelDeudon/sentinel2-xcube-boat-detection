import os
import sys
import json
import warnings
import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
import torch
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from typing import Tuple

def preprocess(cube: Dataset, max_cloud_proba: float = 0.1, nans_how: str = 'any', verbose: int = 1,
               plot_NDWI: bool = True) -> Tuple[Dataset, DataArray]:
    """ Preprocess cube for boat detect.
    
    Args:
        cube: xCube object with time dimension and (B03, B08, CLP) bands or (B03, B08, B04).
        max_cloud_proba: float in [0,1]. Default 0.005 will keep imagery 99.5% cloudless.
        nans_how: str, 'any' or 'all'.
        verbose: int or bool.
        plot_NDWI: bool. Default True will plot the NDWI layer.
    Return:
        cube: xCube object, preprocessed.
    """

    n_snaps = len(cube.time)
    cube = cube.dropna(dim='time',how=nans_how) # drop images w/ any nan
    if verbose:
        print('Keeping {}/{} images without nans'.format(len(cube.time), n_snaps))
        
    if hasattr(cube, 'CLP'):
        cube = cube.where(cube.CLP.mean(dim=('lat','lon'))<255*max_cloud_proba, drop=True) # sentinelhub cloud mask
    elif hasattr(cube, 'B03') and hasattr(cube, 'B04'):
        cloud_mask = ( (cube.B03>0.175)*(cube.B03>cube.B04) + (cube.B03>0.39) )>0 # cloud detector, reference (Braaten-Cohen-Yang, 2015)
        cube = cube.where(cloud_mask.mean(dim=('lat','lon'))<max_cloud_proba, drop=True)
    if verbose:
        print('Keeping {}/{} images {}% cloudless'.format(len(cube.time), n_snaps, 100*(1-max_cloud_proba))) # keep cloudless imagery
    
    ndwi = (cube.B03-cube.B08)/(cube.B03+cube.B08) # NDWI, reference (McFeeters, 1996)
    ndwi.attrs['long_name']='-NDWI'
    ndwi.attrs['units']='unitless'
    cube['NDWI'] = -ndwi # add negative NDWI (high value for non water)
    if plot_NDWI:
        (-cube).NDWI.plot.imshow(col='time', col_wrap=4, cmap='RdYlBu') ##### plot False Color instead!!!
    cube['NDWI'] = (cube.NDWI+1.0)/2.0 # from [-1,1] to [0,1]
    cube = cube*(cube<1.0) # clip other bands to [0,1]
    background_ndwi = cube.NDWI.min(dim='time')
    return cube, background_ndwi


def cube2tensor(cube, max_cloud_proba=0.1, nans_how='any', verbose=1, plot_NDWI=True):
    """ Convert xcube to tensor and metadata"""
    cube, background_ndwi = preprocess(cube, max_cloud_proba=max_cloud_proba, nans_how=nans_how, verbose=1, plot_NDWI=plot_NDWI)
    timestamps = [str(t)[:10] for t in cube.time.values] # format yyyy-mm-dd
    array = np.stack([np.stack([cube.B08.values[t], background_ndwi.values], 0) for t in range(len(timestamps))], 0) # (T,C,H,W)
    #clp = np.stack([np.stack([cube.CLP.values[t]], 0) for t in range(len(timestamps))], 0) # (T,C,H,W)
    x = torch.from_numpy(array)
    #assert array.shape[1] == self.input_dim
    return x, timestamps
    

def plot_cube_and_background(cube, background_ndwi, t=0, figsize=(25,5), cmap='grey'):
    """ Plot a cube and background. 
    Args:
        cube: xCube object with time dimension and (B08,CLP) bands.
        background: xCube object with NDWI bands.
        t: int, time indexé
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)

    plt.subplot(1,3,1)
    cube.B08.isel(time=t).plot(cmap=cmap)
    plt.subplot(1,3,2)
    cube.CLP.isel(time=t).plot(cmap=cmap)
    plt.subplot(1,3,3)
    background_ndwi.plot(cmap=cmap)
    plt.show()
    
    
def save_labels(cube, background_ndwi, label, lat_lon, data_dir='data/chips', label_filename='data/labels.json'):
    """ Save preprocessed imagery and labels to disk.
    Args:
        cube: xCube object with time dimension and (B02,B03,B04,B08,NDWI) bands.
        background: xCube object with (B02,B03,B04,B08,NDWI) bands.
        label: list of int, boat counts for each time stamps.
        lat_lon: tuple of floats, latitude and longitude in degrees.
        data_dir: str, path to chips folder.
        label_filename: str, path to filename of labels dict (for recovery).
    """

    if len(label) != len(cube.time):
        print('Error: Cube and Label have different length')
        return 0
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    os.makedirs(data_dir, exist_ok=True)
    subdir = 'lat_{}_lon_{}'.format(str(lat_lon[0]).replace('.','_'), str(lat_lon[1]).replace('.','_'))
    os.makedirs(os.path.join(data_dir,subdir), exist_ok=True)
    with open(label_filename, 'r') as f:
        label_file = json.load(f)
        if subdir not in label_file:
            label_file[subdir] = {}
    
    # save background + imagery and labels
    imsave(os.path.join(data_dir,subdir,'bg_ndwi.png'), img_as_ubyte(background_ndwi.values))
    for t, y in enumerate(label):
        snap_date = str(cube.isel(time=t).time.values)[:10]
        label_file[subdir][snap_date] = y
        imsave(os.path.join(data_dir,subdir,'img_03_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B03.values))
        imsave(os.path.join(data_dir,subdir,'img_08_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B08.values))
        imsave(os.path.join(data_dir,subdir,'img_clp_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).CLP.values))
    
    with open(label_filename, 'w') as f:
        json.dump(label_file, f)
        print('Saved {} labels for {}'.format(len(label), subdir))
    return 1


def save_cubes(cube, background_ndwi, lat_lon, data_dir='data/chips'):
    """ Save preprocessed imagery to disk.
    Args:
        cube: xCube object with time dimension and (B02,B03,B04,B08,NDWI) bands.
        background: xCube object with (B02,B03,B04,B08,NDWI) bands.
        lat_lon: tuple of floats, latitude and longitude in degrees.
        data_dir: str, path to chips folder.
    """
    import os
    import sys
    import warnings
    from skimage.io import imsave
    from skimage import img_as_ubyte

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    os.makedirs(data_dir, exist_ok=True)
    subdir = 'lat_{}_lon_{}'.format(str(lat_lon[0]).replace('.', '_'), str(lat_lon[1]).replace('.', '_'))
    os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    # save background + imagery and labels
    imsave(os.path.join(data_dir, subdir, 'bg_ndwi.png'), img_as_ubyte(background_ndwi.values))
    for t in cube.time: # y is the account of ships in the image
        snap_date = str(t.values)[:10]

        imsave(os.path.join(data_dir, subdir, 'img_03_t_{}.png'.format(snap_date)),
               img_as_ubyte(cube.sel(time=t).B03.values))
        imsave(os.path.join(data_dir, subdir, 'img_08_t_{}.png'.format(snap_date)),
               img_as_ubyte(cube.sel(time=t).B08.values))
        imsave(os.path.join(data_dir, subdir, 'img_clp_t_{}.png'.format(snap_date)),
               img_as_ubyte(cube.sel(time=t).CLP.values))
        imsave(os.path.join(data_dir, subdir, 'img_ndwi_t_{}.png'.format(snap_date)),
               img_as_ubyte(cube.sel(time=t).NDWI.values))
        print('Saved cubes with timestamp {} under {}'.format(snap_date, subdir))