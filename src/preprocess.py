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
import xarray
from typing import Tuple, List
import matplotlib.pyplot as plt

from GIS_utils import bbox_from_point
from config import CubeConfig
from model import Model

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
        cube = cube.where(cube.CLP.mean(dim=('lat','lon'))<255*max_cloud_proba, drop=True) # sentinelhub cloud mask in [0,255.]
        cube['CLP'] = cube.CLP/255.
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
        cube.NDWI.plot.imshow(col='time', col_wrap=4, cmap='coolwarm') ##### plot False Color instead!!!
    cube['NDWI'] = (cube.NDWI+1.0)/2.0 # from [-1,1] to [0,1]
    cube = cube*(cube<=1.0) + 1.*(cube>1.0) # clip other bands to [0,1]
    background_ndwi = cube.NDWI.min(dim='time')
    return cube, background_ndwi


def cube2tensor(cube, max_cloud_proba=0.1, nans_how='any', verbose=1, plot_NDWI=True, bg_ndwi_path=None):
    """ Convert xcube to tensor and metadata"""
    #TODO add use_cached_bg and stop preprocess bg.
    cube, background_ndwi = preprocess(cube, max_cloud_proba=max_cloud_proba, nans_how=nans_how, verbose=verbose, plot_NDWI=plot_NDWI)
    timestamps = [str(t)[:10] for t in cube.time.values] # format yyyy-mm-dd
    #x = np.stack([np.stack([cube.B08.values[t], background_ndwi.values, cube.CLP.values[t]], 0) for t in range(len(timestamps))], 0) # (T,3,H,W)
    ### TODO load bg from cached files
    if bg_ndwi_path:
        background_ndwi = np.load(bg_ndwi_path)
        x = np.stack([np.stack([cube.B08.values[t], background_ndwi], 0) for t in range(len(timestamps))],
                     0)  # (T,3,H,W)
    else:
        x = np.stack([np.stack([cube.B08.values[t], background_ndwi.values], 0) for t in range(len(timestamps))], 0) # (T,3,H,W)
    x = torch.from_numpy(x)
    return x, timestamps
    

def plot_cube_and_background(cube, background_ndwi, t=0, figsize=(25,5)):
    """ Plot a cube and background. 
    Args:
        cube: xCube object with time dimension and (B08,CLP) bands.
        background: xCube object with NDWI bands.
        t: int, time index
    """
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # gpu support
    model = Model(input_dim=2, hidden_dim=16, kernel_size=3, device=device, version='0.1.1')
    checkpoint_file = os.path.join("../factory", model.folder, 'model.pth')
    model.load_checkpoint(checkpoint_file=checkpoint_file)
    model = model.eval()
    #x = np.stack([np.stack([cube.B08.values[t], background_ndwi.values, cube.CLP.values[t]], 0) for t in range(len(cube.time))], 0) # (T,3,H,W)
    x = np.stack([np.stack([cube.B08.values[t], background_ndwi.values], 0) for t in range(len(cube.time))], 0) # (T,3,H,W)
    x = torch.from_numpy(x)
    heatmaps, counts = model.chip_and_count(x, filter_peaks=True, downsample=True, plot_heatmap=False, plot_indicator=False)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.subplot(1,4,1)
    background_ndwi.plot(cmap='coolwarm')
    plt.subplot(1,4,2)
    cube.B08.isel(time=t).plot(cmap='coolwarm')
    plt.xticks([]); plt.yticks([])
    plt.subplot(1,4,3)
    plt.xticks([]); plt.yticks([])
    plt.imshow(heatmaps[t], cmap='Reds')
    plt.title('y_hat = {:.1f}'.format(counts[t]))
    plt.subplot(1,4,4)
    cube.CLP.isel(time=t).plot(cmap='gray')
    plt.xticks([]); plt.yticks([])
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
    df_labels = pd.read_csv(label_filename, usecols=["lat_lon", "timestamp", "count"])
    labeled_files = []
    for t, y in enumerate(label):
        snap_date = str(cube.isel(time=t).time.values)[:10]
        imsave(os.path.join(data_dir,subdir,'img_03_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).B03.values))
        imsave(os.path.join(data_dir,subdir,'img_08_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).B08.values))
        imsave(os.path.join(data_dir,subdir,'img_clp_t_{}.png'.format(snap_date)), img_as_ubyte(cube.isel(time=t).CLP.values))
        labeled_files.append((subdir, snap_date, y))
    df1 = pd.DataFrame(labeled_files).rename(columns={0: "lat_lon", 1: "timestamp",  2: "count"})
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


def remove_s1_empty_nans(cube: Dataset, nans_how='any'):
    """
    remove any bands that doesn't exist in cube and time stamps that has no
    :param cube:
    :param nans_how:
    :return:
    """
    all_bands = ["HH", "HV", "VH", "VV", "HH+HV", "VV+VH"]
    for band in all_bands:
        try:
            cube[band].values
            cube[band] = cube[band].dropna(dim='time', how=nans_how)
            cube = cube.where(cube[band].mean(dim=('lat', 'lon')) > 0.0, drop=True)
        except:
            cube = cube.drop_vars(band)
    return cube


def generate_bg_from_s1(cube: Dataset, bg_band="VV", fused_by="min"):
    """
    :param cube: xarray dataset
    :param bg_band: from which band the background should be generated
    :param fused_by: either min, max, median or mean
    :return: the backgound xarray
    """
    if fused_by == "min":
        background = cube[bg_band].min(dim="time")
    elif fused_by == "mean":
        background = cube[bg_band].mean(dim="time")
    elif fused_by == "median":
        background = cube[bg_band].median(dim="time")
    else:
        background = cube[bg_band].max(dim="time")
    return background

def generate_landwater_mask(cube:Dataset, threshold_quantile=0.625, plot=True):
    """
        generate a binary land water mask of given cube
    :param cube: xarray Dataset
    :param plot: if to plot
    :param threshold_quantile:
    :return: a binary land water mask
    """
    bg_VV = generate_bg_from_s1(cube, bg_band="VV", fused_by="min")
    bg_VH = generate_bg_from_s1(cube, bg_band="VH", fused_by="min")
    threshold_VV = bg_VV.quantile(q=threshold_quantile)
    threshold_VH = bg_VH.quantile(q=threshold_quantile)
    binary_bg_VH = bg_VH.where(bg_VH > threshold_VH).fillna(1).where(bg_VH <= threshold_VH).fillna(0)
    binary_bg_VV = bg_VV.where(bg_VV > threshold_VV).fillna(1).where(bg_VV <= threshold_VV).fillna(0)
    binary_bg = binary_bg_VH * binary_bg_VV
    if plot:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        binary_bg_VH.plot.imshow()
        plt.title("binary_bg_VH")
        plt.subplot(1, 3, 2)
        binary_bg_VV.plot.imshow()
        plt.title("binary_bg_VV")
        plt.subplot(1, 3, 3)
        binary_bg.plot.imshow()
        plt.title("binary_bg")
    stacked_binary_bg_VH = xarray.concat([binary_bg_VH.expand_dims('time') for i in range(len(cube.time))], dim='time')
    stacked_binary_bg_VV = xarray.concat([binary_bg_VV.expand_dims('time') for i in range(len(cube.time))], dim='time')
    stacked_binary_bg = xarray.concat([binary_bg.expand_dims('time') for i in range(len(cube.time))], dim='time')

    return stacked_binary_bg_VH, stacked_binary_bg_VV, stacked_binary_bg