import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from xcube_sh.cube import open_cube

from src.GIS_utils import bbox_from_point
from src.config import CubeConfig
from src.preprocess import cube2tensor
from src.model import Model


def load_model(checkpoint_dir="/home/jovyan/checkpoints", version="0.0.4"):
    """
    Args:
        checkpoint_dir: str, path to checkpoint directory
        version: str, example '0.0.1'
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # gpu support
    model = Model(input_dim=2, hidden_dim=16, kernel_size=3, device=device, version=version)
    checkpoint_file = os.path.join(checkpoint_dir, model.folder, 'model.pth')
    model.load_checkpoint(checkpoint_file=checkpoint_file)
    model = model.eval()
    print('Loaded model version {} on {}'.format(version, device))
    return model


##### Cache results
def coords2counts(model, coords, time_window, radius=5000, time_period='5D', max_cloud_proba=0.2):
    '''
    Args:
        model: pytorch model
        coords: list or tuple, lat and lon.
        time_window: list or tuple, start and end dates.
        radius: int
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
    Returns:
        traffic: dict, timestamps and counts
    '''
    lat, lon = coords[0], coords[1]    
    bbox = bbox_from_point(lat=lat, lon=lon, r=radius) # WGS84 coordinates
    cube_config = CubeConfig(dataset_name='S2L1C', band_names=['B03', 'B08', 'CLP'], tile_size=[2*radius//10, 2*radius//10], geometry=bbox, time_range=time_window, time_period=time_period,)
    cube = open_cube(cube_config, max_cache_size=2**30)
    x, timestamps = cube2tensor(cube, max_cloud_proba=max_cloud_proba, nans_how='any', verbose=0, plot_NDWI=False) # Convert Cube to tensor (NIR + BG_NDWI) and metadata.
    heatmaps, counts = model.chip_and_count(x, water_ndwi=0.5, plot_heatmap=True, timestamps=timestamps, max_frames=6, plot_indicator=True, filter_peaks=True, shift_pool=True) # Detect and count boats!

    ##### Save AOI, timestamps, counts to geodB
    traffic = {}
    for timestamp, count in zip(timestamps, counts):
        traffic[timestamp] = float(count)
    return traffic


def scan_AOI(interest='Straits', time_windows=[['2019-01-01', '2019-05-28'], ['2020-01-01', '2020-05-28']], data_dir="/home/jovyan/data", checkpoint_dir="/home/jovyan/checkpoints", version="0.0.4", radius=5000, time_period='5D', max_cloud_proba=0.2):
    '''
    Args:
        interest: str, 'Straits' or 'Ports'
        time_windows: list
        data_dir: str, path to data directory
        checkpoint_dir: str, path to checkpoint directory
        version: str, example '0.0.1'
        radius: int, AOI radius in meters
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
    Returns:
        boat_traffic: dict, AOI and traffic
    '''

    model = load_model(checkpoint_dir, version) ### load pretrained model
    
    with open(os.path.join(data_dir, 'aoi.json'), 'r') as f:
        aoi_file = json.load(f)[interest] ### load AOI
    
    boat_traffic = {}
    for aoi_name in aoi_file.keys(): # loop ovr AOI
        coords = aoi_file[aoi_name][0]
        query = '{}_lat={}_lon={}_r={}_v={}'.format(aoi_name.lower(), coords[0], coords[1], radius, version) ### Compare 2019/2020 counts
        print(query.capitalize())
        output_filename = os.path.join(data_dir, 'outputs', '{}.json'.format(query))
        if not os.path.exists(output_filename):
            for time_window in time_windows:
                try:
                    traffic = coords2counts(model=model, coords=coords, radius=radius, time_window=time_window, time_period=time_period, max_cloud_proba=max_cloud_proba) # configure, download, preprocess cube
                    if query in boat_traffic:
                        boat_traffic[query].append(traffic)
                    else:
                        boat_traffic[query] = [traffic]
                except:
                    pass

            if query in boat_traffic:
                with open(output_filename, 'w+') as f:
                    json.dump(boat_traffic[query], f, sort_keys=True, indent=4)

        else:
            with open(output_filename, 'r') as f:
                boat_traffic[query] = json.load(f)
    return boat_traffic
     
    
def aggregate_counts(counts, timestamps):
    ''' ##### TODO: Aggregate by week day'''
    agg_counts, agg_timestamps = {}, []
    current_month_id = '00'
    for count, timestamp in zip(counts, timestamps):
        month_id = timestamp[2:9]
        if current_month_id != month_id:
            current_month_id = month_id
            agg_counts[month_id] = [count]
            agg_timestamps.append(month_id)
        else:
            agg_counts[month_id].append(count)
    agg_counts = [np.mean(count_samples) for count_samples in agg_counts.values()]
    return agg_counts, agg_timestamps
    
    
def analyze_boat_traffic(boat_traffic, kernel_size=3):
    '''
    Args:
        boat_traffic: dict, AOI and traffic
        kernel_size: int, median filtering (noisy signal)
    '''
    
    for query in boat_traffic.keys():

        try: #####
            timestamps0 =list(boat_traffic[query][0].keys())
            counts0 = list(boat_traffic[query][0].values())
            counts0 = list(medfilt(counts0, kernel_size=kernel_size))
            #counts0, timestamps0 = aggregate_counts(counts0, timestamps0)
            timestamps1 =list(boat_traffic[query][1].keys())
            counts1 = list(boat_traffic[query][1].values())
            counts1 = list(medfilt(counts1, kernel_size=kernel_size))
            #counts1, timestamps1 = aggregate_counts(counts1, timestamps1)
            ymin, ymax = min(counts0+counts1), max(counts0+counts1)

            
            delta = 100*(np.mean(counts1)-np.mean(counts0))/np.mean(counts0) ###

            plt.figure(1, figsize=(30,5))
            plt.subplot(121)
            plt.plot(timestamps0, counts0, color='gray', marker='o')
            plt.plot(timestamps1, counts1, color='black', marker='+')
            #plt.xticks(list(range(len(timestamps0))), timestamps0, rotation=45)
            #plt.xticks(list(range(len(timestamps1))), timestamps1, rotation=45)
            plt.ylim(ymin, ymax)
            plt.title(query+'\n Delta 2019/2020: {:.1f}%'.format(delta))
            plt.show()

        except:
            pass

        
        
        
if __name__ == '__main__':
    
    ##### Import argparse --> CLI
    ##### Automate analysis for AOI + time window  
    
    boat_traffic = scan_AOI(interest='Straits', time_windows=[['2019-01-01', '2019-05-28'], ['2020-01-01', '2020-05-28']], version="0.0.4", radius=5000, time_period='5D', max_cloud_proba=0.2)
    analyze_boat_traffic(boat_traffic, kernel_size=3)
