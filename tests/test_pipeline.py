''' Test sentinel2-xcube-boat-detection. '''

import os
import sys
# sys.path.insert(0,os.path.dirname('src/'))


def test_imports():
    ''' Test imports and dependencies'''
    import json
    import geopandas as pd
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import pandas as pd
    import plotly
    import shapely
    import skimage
    import sklearn
    import torch
    import tqdm
    import typing


def test_GIS_utils():
    ''' Test GIS util to get squared bounding box from center (degrees) and radius (meters).'''
    import numpy as np
    from src.GIS_utils import bbox_from_point
    x1, y1, x2, y2 = bbox_from_point(lat=40.0482, lon=26.3013, r=1000)
    assert isinstance(x1, float) and isinstance(x2, float) and isinstance(y1, float) and isinstance(y2, float)
    assert np.abs(x1-x2) < 1.0 and np.abs(y1-y2) < 1.0 # difference smaller than 1 degree


def test_model():
    ''' Test pytorch model to detect and count objects in sentinel-2 imagery'''
    import torch
    from src.model import Model

    batch_size, input_dim, H, W = 8, 2, 100, 100
    x = torch.zeros((batch_size, input_dim, H, W))
    y = torch.ones((batch_size, 1))

    hidden_dim, pool_size, n_max = 16, 10, 1
    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=3, pool_size=pool_size, n_max=n_max, pad=True, device='cpu', version='0.0.2')
    pixel_embedding, density_map, p_hat, y_hat = model(x, water_ndwi=-1.0, downsample=True)
    assert pixel_embedding.shape == (batch_size, hidden_dim, H, W) # pixel embedding
    assert density_map.shape == (batch_size, 1, H//pool_size, W//pool_size) # density map
    assert p_hat.shape == y.shape # proba boat presence
    assert y_hat.shape == y.shape # boat counts (expectation)

    metrics = model.get_loss(x, y, n=None, ld=0.2, water_ndwi=-1.0)
    for metric in ['loss', 'clf_error', 'reg_error', 'accuracy', 'precision', 'recall', 'f1']:
        assert metric in metrics.keys()

    heatmaps, counts = model.chip_and_count(x, water_ndwi=0.5)
    assert len(heatmaps) == batch_size and heatmaps[0].shape == (H//pool_size, W//pool_size)
    assert len(counts) == batch_size and isinstance(counts[0], float)


def test_checkpoint():
    ''' Load checkpoint for pretrained model'''
    from src.model import Model
    model = Model(input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, pad=True, device='cpu', version='0.0.1')
    #model.load_checkpoint(checkpoint_file)


def test_CubeConfig():
    ''' Test xcube_sh config'''
    from src.config import CubeConfig


def test_preprocess():
    ''' Test utils to preprocess sentinel2-cube'''
    from src.preprocess import preprocess
    import numpy as np
    from xarray.core.dataset import Dataset
    import pandas as pd
    t = 5
    B03 = np.random.rand(t, 100, 100)
    B08 = np.random.rand(t, 100, 100)
    CLP = np.zeros((t, 100, 100))

    lon = np.repeat(-99.83, 100).tolist()
    lat = np.repeat(42.25, 100).tolist()

    cube = Dataset({'B03':(["time", 'lat', 'lon'], B03),
             'B08': (["time", 'lat', 'lon'], B08),
             'CLP': (["time", 'lat', 'lon'], CLP)},
             coords = {'lon': lon, 'lat': lat,
                       'time': pd.date_range('2014-09-06', periods=t),
                       'reference_time': pd.Timestamp('2014-09-05')}
            )

    cube, background_ndwi = preprocess(cube, max_cloud_proba=0.1, nans_how='any', verbose=1, plot_NDWI=False)
    assert hasattr(cube, "NDWI")
    # assert hasattr(background_ndwi, "NDWI")
    assert background_ndwi.name == "NDWI"
    assert background_ndwi.data.shape == (100, 100)



        
def test_dataset():
    ''' Test pytorch.Dataset to train model'''
    #from dataset import plot_geoloc, getImageSetDirectories, S2_Dataset
    #s2_dataset = S2_Dataset(imset_dir=[], augment=True)
    #assert len(s2_dataset) == 0
    pass


def test_train():
    ''' Test train script'''
    #from train import train, get_failures_or_success
    pass
