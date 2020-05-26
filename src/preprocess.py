import os
import sys
import json
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from skimage.io import imsave
from skimage import img_as_ubyte
import torch
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray

from typing import Tuple, List
from src.GIS_utils import bbox_from_point
from src.config import CubeConfig

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
    cube, background_ndwi = preprocess(cube, max_cloud_proba=max_cloud_proba, nans_how=nans_how, verbose=verbose, plot_NDWI=plot_NDWI)
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
    
    
def save_labels(cube, background_ndwi, label, lat_lon, data_dir='data/chips', label_filename='data/labels.csv'):
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

    # save background + imagery and labels
    imsave(os.path.join(data_dir,subdir,'bg_ndwi.png'), img_as_ubyte(background_ndwi.values))
    df_labels = pd.read_csv(label_filename)
    labeled_files = []
    for t, y in enumerate(label):
        snap_date = str(cube.isel(time=t).time.values)[:10]
        imsave(os.path.join(data_dir,subdir,'img_03_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).B03.values))
        imsave(os.path.join(data_dir,subdir,'img_08_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).B08.values))
        imsave(os.path.join(data_dir,subdir,'img_clp_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).CLP.values))
        labeled_files.append((os.path.join(data_dir,subdir,'img_08_t_{}.png'.format(snap_date)), y))
    df1 = pd.DataFrame(labeled_files).rename(columns={0: "file_path", 1: "count"})
    #for filename in df1['file_path'].values:
    #    if (filename in df_labels['file_path'].values):
    #        df_labels = df_labels[df_labels['file_path']!=filename]
    df_labels = df_labels.append(df1, ignore_index=True)
    df_labels.to_csv(label_filename, index=False)
    print('Saved {} labels for {}'.format(len(label), subdir))


def save_cubes(cube, background_ndwi, lat_lon, data_dir='data/chips', verbose = True):
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
        if verbose:
            print('Saved cubes with timestamp {} under {}'.format(snap_date, subdir))


def request_save_cubes(start_date: str, end_date: str, lat: float, lon: float, data_chips_dir: str,
                       RADIUS: int = 500, dataset_name: str = 'S2L1C', band_names: List = ['B03', 'B08', 'CLP'],
                       max_cloud_proba: float = 0.1, time_period: str = '1D'):
    """

    :param start_date: '2019-01-01'
    :param end_date: '2019-06-30'
    :param lat: latitude
    :param lon: longitude
    :param data_chips_dir: download image will be saved to this dir like 'data/chips'
    :param RADIUS: radius in meter
    :param dataset_name: either S2L1C or S2L2A
    :param band_names: a list of bands to be saved
    :param max_cloud_proba: maximum probability of cloud
    :param time_period: 1D
    :return:
    """
    from xcube_sh.cube import open_cube
    from xcube_sh.observers import Observers
    bbox = bbox_from_point(lat=lat, lon=lon, r=RADIUS)
    cube_config = CubeConfig(dataset_name=dataset_name,
                             band_names=band_names,  # GREEN + NIR + Clouds
                             tile_size=[2 * RADIUS // 10, 2 * RADIUS // 10],
                             geometry=bbox,
                             time_range=[start_date, end_date],
                             time_period=time_period,
                             )
    request_collector = Observers.request_collector()
    cube = open_cube(cube_config, observer=request_collector)

    cube, background_ndwi = preprocess(cube, max_cloud_proba=max_cloud_proba,
                                       nans_how='any', verbose=1, plot_NDWI=False)
    save_cubes(cube, background_ndwi, lat_lon=(lat, lon), data_dir=Path(data_chips_dir), verbose=False)
