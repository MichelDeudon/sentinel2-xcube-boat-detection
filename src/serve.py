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

### load pretrained model
checkpoint_dir = "/home/jovyan/checkpoints"
version = "0.0.1"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # gpu support
model = Model(input_dim=2, hidden_dim=16, kernel_size=3, device=device, version=version)
checkpoint_file = os.path.join(checkpoint_dir, model.folder, 'model.pth')
model.load_checkpoint(checkpoint_file=checkpoint_file)
model = model.eval()
print('Model version: v{}'.format(version))
print('Device: {} \n'.format(device))

### load predefined AOI and time windows
data_dir = "/home/jovyan/data" # data directory (path)
RADIUS = 5000 # AOI radius in meters
max_cloud_proba = 0.1 # max cloud coverage
with open(os.path.join(data_dir, 'aoi.json'), 'r') as f:
    aoi_file = json.load(f)['Straits']
    aoi_names = [name for name in aoi_file.keys()]
time_window1 = ['2019-01-01', '2019-05-15']
time_window2 = ['2020-01-01', '2020-05-15']
time_period = '5D'
print('AOI: {}'.format(', '.join(aoi_names)))
print('Radius: {}m'.format(RADIUS))
print('Cloud coverage (max): {}%'.format(100*max_cloud_proba))
print('Time window 1: {} / {}'.format(time_window1[0],time_window1[1]))
print('Time window 2: {} / {}'.format(time_window2[0],time_window2[1]))
print('Time period: {}\n'.format(time_period))

##### Cache results
def coords2counts(coords, radius, time_window, time_period, max_cloud_proba, water_ndwi=0.5, outliers=500):
    '''
    Args:
        coords: list or tuple, lat and lon.
        radius: int
        time_window: list or tuple, start and end dates.
        time_period: str, example '5D'
        max_cloud_proba: float, max cloud coverage
        water_ndwi: float, NDWI threshold for water/land segmentation
        outliers: int, max number of counts
    Returns:
        traffic: dict, timestamps and counts
    '''
    lat, lon = coords[0], coords[1]    
    bbox = bbox_from_point(lat=lat, lon=lon, r=radius) # WGS84 coordinates
    cube_config = CubeConfig(dataset_name='S2L1C', band_names=['B03', 'B08', 'CLP'], tile_size=[2*radius//10, 2*radius//10], geometry=bbox, time_range=time_window, time_period=time_period,)
    cube = open_cube(cube_config, max_cache_size=2**30)
    x, timestamps = cube2tensor(cube, max_cloud_proba=max_cloud_proba, nans_how='any', verbose=0, plot_NDWI=False) # Convert Cube to tensor (NIR + BG_NDWI) and metadata.
    heatmaps, counts = model.chip_and_count(x, water_ndwi=water_ndwi, plot_heatmap=True, timestamps=timestamps, max_frames=8, plot_indicator=False, outliers=outliers) # Detect and count boats!

    ##### Save AOI, timestamps, counts to geodB
    traffic = {}
    for timestamp, count in zip(timestamps, counts):
        traffic[timestamp] = float(count)
    return traffic




### loop over aoi (configure, download, preprocess cube)
boat_traffic = {} # Density: Bosporus > Dardanelles > Gibraltar > Dover ### Suez, Korinth, Otranto, Messina, Grande Belt, Kiel Canal, Tallin-Helsinki
#aoi_names = ['Bosporus'] ###### COMMENT
for aoi_name in aoi_names:
    coords = aoi_file[aoi_name][0]
    query = '{}_lat={}_lon={}_r={}_v={}'.format(aoi_name.lower(), coords[0], coords[1], RADIUS, version) ### Compare 2019/2020 counts
    output_filename = os.path.join(data_dir, 'outputs', '{}.json'.format(query))
    if not os.path.exists(output_filename):
        for time_window in [time_window1, time_window2]:
            traffic = coords2counts(coords, radius=RADIUS, time_window=time_window, time_period=time_period, max_cloud_proba=max_cloud_proba)
            if query in boat_traffic:
                boat_traffic[query].append(traffic)
            else:
                boat_traffic[query] = [traffic]
        with open(output_filename, 'w+') as f:
            json.dump(boat_traffic[query], f, sort_keys=True, indent=4)
    else:
        with open(output_filename, 'r') as f:
            boat_traffic[query] = json.load(f)
        
    #break #####
    ##### Automate analysis for AOI + time window
    
    ##### Import argparse --> CLI
    
    
def aggregate_counts(counts, timestamps):
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
    

kernel_size = 3 # median filtering (noisy signal)
for query in boat_traffic.keys():
    
    
    timestamps0 =list(boat_traffic[query][0].keys())
    counts0 = list(boat_traffic[query][0].values())
    counts0 = list(medfilt(counts0, kernel_size=kernel_size))
    counts0, timestamps0 = aggregate_counts(counts0, timestamps0)
    timestamps1 =list(boat_traffic[query][1].keys())
    counts1 = list(boat_traffic[query][1].values())
    counts1 = list(medfilt(counts1, kernel_size=kernel_size))
    counts1, timestamps1 = aggregate_counts(counts1, timestamps1)
    ymin, ymax = min(counts0+counts1), max(counts0+counts1)
    
    ###
    delta = 100*(np.mean(counts1)-np.mean(counts0))/np.mean(counts0)
    if delta>0:
        color1 = 'green'
    else:
        color1 = 'red'

    plt.figure(1, figsize=(30,5))
    plt.subplot(121)
    plt.plot(timestamps0, counts0, color='gray', marker='o')
    plt.plot(timestamps1, counts1, color=color1, marker='+')
    #plt.xticks(list(range(len(timestamps0))), timestamps0, rotation=45)
    #plt.xticks(list(range(len(timestamps1))), timestamps1, rotation=45)
    plt.ylim(ymin, ymax)
    plt.title(query+'\n Delta 2019/2020: {:.1f}%'.format(delta))
    plt.show()