import os
import json
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.signal import medfilt
from xcube_sh.cube import open_cube

from GIS_utils import bbox_from_point
from config import CubeConfig
from preprocess import cube2tensor, preprocess
from model import Model


def load_model(checkpoint_dir="../factory", version="0.1.1"):
    """
    Args:
        checkpoint_dir: str, path to checkpoint directory
        version: str, example '0.0.1'
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # gpu support
    model = Model(input_dim=2, hidden_dim=16, kernel_size=3, device=device, version=version) #####
    checkpoint_file = os.path.join(checkpoint_dir, model.folder, 'model.pth')
    model.load_checkpoint(checkpoint_file=checkpoint_file)
    model = model.eval()
    print('Loaded model version {} on {}'.format(version, device))
    return model

def coords2counts(model, coords, time_window, time_period='5D', max_cloud_proba=0.2,
                  use_cached_bg_ndwi=False, bg_ndwi_dir="data/chips/"):
    '''
    Args:
        model: pytorch model
        coords: list or tuple, lat and lon.
        time_window: list or tuple, start and end dates.
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
    Returns:
        traffic: dict, timestamps and counts
    '''
    lat, lon, radius = coords[0], coords[1], coords[2]
    bbox = bbox_from_point(lat=lat, lon=lon, r=radius) # WGS84 coordinates
    cube_config = CubeConfig(dataset_name='S2L1C', band_names=['B03', 'B08', 'CLP'], tile_size=[2*radius//10, 2*radius//10], geometry=bbox, time_range=time_window, time_period=time_period,)
    cube = open_cube(cube_config, max_cache_size=2**30)
    #TODO add use_cached_bg to cube2tensor to stop processing bg image
    #TODO bg calculated using 3 month's data, can be updated every month
    if use_cached_bg_ndwi:
        subdir = 'lat_{}_lon_{}'.format(str(lat).replace('.', '_'), str(lon).replace('.', '_'))
        bg_ndwi_path = os.path.join(bg_ndwi_dir, subdir, "bg_ndwi.png")
        x, timestamps = cube2tensor(cube, max_cloud_proba=max_cloud_proba, nans_how='any', verbose=0,
                                    plot_NDWI=False, bg_ndwi_path=bg_ndwi_path)  # Convert Cube to tensor (NIR + BG_NDWI) and metadata.
    else:
        x, timestamps = cube2tensor(cube, max_cloud_proba=max_cloud_proba, nans_how='any', verbose=0, plot_NDWI=False) # Convert Cube to tensor (NIR + BG_NDWI) and metadata.
    heatmaps, counts = model.chip_and_count(x, filter_peaks=True, downsample=True, water_NDWI=0.4, plot_heatmap=True, timestamps=timestamps, plot_indicator=True) # Detect and count boats!

    ##### Save AOI, timestamps, counts to geodB. Cache Results.
    traffic = OrderedDict()
    for timestamp, count in zip(timestamps, counts):
        traffic[timestamp] = float(count)
    return traffic


def save_results(boat_traffic_dict, output_filename, model, coords, time_windows, time_period='5D', max_cloud_proba=0.2,
                 use_cached_bg_ndwi=False, bg_ndwi_dir="data/chips/"):
    '''
    Args:
        boat_traffic_dict: dict, traffic in a given AOI
        output_filename: str, path to output file
        model: pytorch model
        coords: list or tuple, lat and lon.
        time_windows: list of (list or tuple), start and end dates.
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
    Returns:
        traffic: dict, timestamps and counts
    '''
    for time_window in time_windows:
        traffic = coords2counts(model, coords, time_window, time_period=time_period, max_cloud_proba=max_cloud_proba,
                                use_cached_bg_ndwi=use_cached_bg_ndwi, bg_ndwi_dir=bg_ndwi_dir) # configure, download, preprocess cube
        boat_traffic_dict = {**boat_traffic_dict, **traffic} # merge dict
    with open(output_filename, 'w+') as f:
        json.dump(boat_traffic_dict, f, sort_keys=True, indent=4) # write file

def save_bg_ndwi(interest='Straits', time_window=['2019-01-01', '2019-05-28'],
                 data_dir="../data", bg_ndwi_dir="data/chips/",
                 time_period='5D', max_cloud_proba=0.2,
                 override=False):
    with open(os.path.join(data_dir, 'aoi.json'), 'r') as f:
        aoi_file = json.load(f)[interest]  ### load AOI (lat, lon, radius)

    for aoi_name in aoi_file.keys():  # loop ovr AOI
        coords = aoi_file[aoi_name][0]
        lat, lon, radius = coords[0], coords[1], coords[2]
        bbox = bbox_from_point(lat=lat, lon=lon, r=radius)  # WGS84 coordinates
        cube_config = CubeConfig(dataset_name='S2L1C', band_names=['B03', 'B08', 'CLP'],
                                 tile_size=[2 * radius // 10, 2 * radius // 10], geometry=bbox, time_range=time_window,
                                 time_period=time_period, )
        cube = open_cube(cube_config, max_cache_size=2 ** 30)
        
        subdir = 'lat_{}_lon_{}'.format(str(lat).replace('.', '_'), str(lon).replace('.', '_'))
        bg_ndwi_path = os.path.join(bg_ndwi_dir, subdir, "bg_ndwi.png")
        if not os.path.isfile(bg_ndwi_path) or override:
            preprocess(cube, max_cloud_proba=max_cloud_proba, nans_how='any',
                                               verbose=0, plot_NDWI=False, bg_ndwi_path=bg_ndwi_path)


def scan_AOI(interest='Straits', time_windows=[['2019-01-01', '2019-05-28'], ['2020-01-01', '2020-05-28']],
             data_dir="../data", checkpoint_dir="../factory", version="0.1.0", time_period='5D', max_cloud_proba=0.2,
             override=False, use_cached_bg_ndwi=False, bg_ndwi_dir="data/chips/"):
    '''
    Args:
        interest: str, 'Straits' or 'Ports'
        time_windows: list
        data_dir: str, path to data directory
        checkpoint_dir: str, path to checkpoint directory
        version: str, example '0.0.1'
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
        override: bool, override results
    Returns:
        boat_traffic: dict, AOI and traffic
    '''

    model = load_model(checkpoint_dir, version) ### load pretrained model
    
    with open(os.path.join(data_dir, 'aoi.json'), 'r') as f:
        aoi_file = json.load(f)[interest] ### load AOI (lat, lon, radius)
    
    boat_traffic = {}
    for aoi_name in aoi_file.keys(): # loop ovr AOI
        coords = aoi_file[aoi_name][0]
        query = '{}_lat={}_lon={}_r={}_v={}'.format(aoi_name.lower(), coords[0], coords[1], coords[2], version) ### Compare 2019/2020 counts
        print(query.capitalize())
        output_filename = os.path.join(data_dir, 'outputs', '{}.json'.format(query))
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                boat_traffic[query] = json.load(f) # read file
        else:
            boat_traffic[query] = OrderedDict()
            
        if override or not os.path.exists(output_filename):
            save_results(boat_traffic[query], output_filename, model, coords, time_windows,
                         time_period=time_period, max_cloud_proba=max_cloud_proba,
                         use_cached_bg_ndwi=use_cached_bg_ndwi, bg_ndwi_dir=bg_ndwi_dir)
            
    return boat_traffic
     
    
def aggregate_counts_by_months(counts, timestamps):
    '''
    Args:
        counts: list
        timestamps: list
    Returns:
        agg_counts: list
        agg_timestamps: list
    '''
    agg_counts, agg_timestamps = {}, []
    current_year_month = '0000-00'
    for timestamp, count in zip(timestamps, counts):
        year_month = timestamp[:7] # 2019-12-31
        if current_year_month != year_month:
            current_year_month = year_month
            agg_counts[current_year_month] = [count]
            agg_timestamps.append(current_year_month)
        else:
            agg_counts[current_year_month].append(count)
    agg_counts = [np.mean(count_samples) for count_samples in agg_counts.values()]
    return agg_counts, agg_timestamps

def filter_by_weekday(counts, timestamps, week_day=0):
    '''
    Args:
        counts: list
        timestamps: list
    Returns:
        new_counts: list
        new_timestamps: list
    '''
    new_counts, new_timestamps = [], []
    for timestamp, count in zip(timestamps, counts):
        if datetime.datetime(*[int(item) for item in timestamp.split('-')]).weekday() == week_day:
            new_counts.append(count)
            new_timestamps.append(timestamp)
    return new_counts, new_timestamps
    
    
def analyze_boat_traffic(boat_traffic, kernel_size=3, week_day=0, aggregate_by_month=True, deltas=[ [['2019-01-01','2019-06-01'], ['2020-01-01','2020-06-01']]]):
    '''
    Args:
        boat_traffic: dict, AOI and traffic
        kernel_size: int, median filtering (noisy signal)
        week_day: int
        aggregate_by_month: bool
    '''
    
    for query in boat_traffic.keys():
        timestamps, counts = [], []
        traffic = boat_traffic[query]
        timestamps += list(traffic.keys())
        counts += list(traffic.values())
        if week_day is not None:
            counts, timestamps = filter_by_weekday(counts, timestamps, week_day=week_day) # filter time series by week day
            
            
        counts = list(medfilt(counts, kernel_size=kernel_size)) # 1D median filtering to reduce noise
        #counts_median = list(medfilt(counts, kernel_size=kernel_size)) # 1D median filtering to reduce noise
        #new_counts, new_timestamps = [], []
        #for i, (c, t) in enumerate(zip(counts, timestamps)):
        #    ratio = counts_median[i]/np.maximum([c],[1.])
        #    if c==0 or (ratio>=0.5):
        #       new_counts.append(counts_median[i])
        #        new_timestamps.append(t)
        #counts, timestamps = new_counts, new_timestamps
        
        
        if aggregate_by_month:
            counts, timestamps = aggregate_counts_by_months(counts, timestamps) # aggregate by month or time window (trimester)
            
        ### compute "delta" between 2019-2020 + statistics over time windows (median, std, early/end trend, etc.)
        for delta in deltas:
            q1, q2 = [], []
            for timestamp, count in zip(timestamps, counts):
                if timestamp >= delta[0][0] and timestamp <= delta[0][1]:
                    q1.append(count)
                if timestamp >= delta[1][0] and timestamp <= delta[1][1]:
                    q2.append(count)
            q1, q2 = np.mean(q1), np.mean(q2)
            print('{} :: {} - {} / {} - {}: {:.2f}%'.format(query, delta[0][0], delta[0][1], delta[1][0], delta[1][1], 100*(q2-q1)/q1))

        
        # find idx for 2019/2020 split
        idx_2018, idx_2019, idx_2020 = -1, -1, -1
        for idx, t in enumerate(timestamps):
            if t.startswith('2018') and idx_2018<0:
                idx_2018 = idx
            if t.startswith('2019') and idx_2019<0:
                idx_2019 = idx
            if t.startswith('2020') and idx_2020<0:
                idx_2020 = idx
                break
                
        # rename timestamps: remove year to for visualization
        if aggregate_by_month:
            timestamps = [int(t.split('-')[-1]) for t in timestamps] # month only (int)
        else: # month + date/31
            timestamps = [float(t.split('-')[-2]) + float(t.split('-')[-1])/31.0 for t in timestamps] # month only (float)
            
        
        plt.figure(1, figsize=(30,5))
        plt.subplot(121)
        #plt.plot(timestamps[idx_2018:idx_2019], counts[idx_2018:idx_2019], color='gray', marker="1", linestyle='--', label='2018')
        plt.plot(timestamps[idx_2019:idx_2020], counts[idx_2019:idx_2020], color='gray', marker='o', linestyle='-.', label='2019')
        plt.plot(timestamps[idx_2020:], counts[idx_2020:], color='black', marker='+', label='2020')
        plt.legend()
        plt.ylim(bottom=0)
        plt.title(query)
        plt.show()

        
        
        
if __name__ == '__main__':
    
    ##### Import argparse --> CLI
    ##### Automate analysis for AOI + time window  
    boat_traffic = scan_AOI(interest='Straits', time_windows=[['2019-01-01', '2019-05-28'], ['2020-01-01', '2020-05-28']], version="0.1.0", time_period='5D', max_cloud_proba=0.2, override=False)
    analyze_boat_traffic(boat_traffic, kernel_size=1, week_day=0, aggregate_by_month=False)
