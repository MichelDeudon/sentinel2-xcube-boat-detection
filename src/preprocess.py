def preprocess(cube, max_cloud_proba=0.1, nans_how='any', verbose=1, plot_NDWI=True):
    """ Preprocess cube for boat detect.
    
    Args:
        cube: xCube object with time dimension and CLP band or (B03, B04) bands + (B03, B08) bands or more.
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
    cube['NDWI']= -ndwi # add negative NDWI (high value for non water)
    if plot_NDWI:
        (-cube).NDWI.plot.imshow(col='time', col_wrap=4, cmap='RdYlBu') ##### plot False Color instead!!!
        
    #water_land_segmentation = (cube.NDWI.min(dim='time')<-NDWI_threshold) # segment Water/Land using "background" negative NDWI
    #if plot_segmentation:
    #    water_land_segmentation.plot(cmap='gray')
    #cube = cube-(water_land_segmentation*cube).median(dim=('lat','lon')) # augment contrast (substract median value approx. sea) 
    #cube = cube*(cube>0) # keep positive signals (boats generally reflect more than water in the bands considered)
    #cube = cube*water_land_segmentation # mask land 
    #cube -= cube.min(dim=('lat','lon')) # minMaxNormalize
    #cube /= cube.max(dim=('lat','lon'))
    #background = cube.min(dim='time') # background image
    cube['NDWI'] = (cube.NDWI+1.0)/2.0 # from [-1,1] to [0,1]
    cube = cube*(cube<1.0) # clip other bands to [0,1]
    background = cube.min(dim='time')
    
    return cube, background
    

def plot_cube_and_background(cube, background, t=0):
    """ Plot a cube and background. 
    Args:
        cube: xCube object with time dimension and (B02,B03,B04,B08,NDWI) bands.
        background: xCube object with (B02,B03,B04,B08,NDWI) bands.
        t: int, time indexé
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16,20))
    plt.subplot(5,2,1)
    cube.B02.isel(time=t).plot(cmap='gray')
    plt.subplot(5,2,2)
    background.B02.plot(cmap='gray')
    plt.subplot(5,2,3)
    cube.B03.isel(time=t).plot(cmap='gray')
    plt.subplot(5,2,4)
    background.B03.plot(cmap='gray')
    plt.subplot(5,2,5)
    cube.B04.isel(time=t).plot(cmap='gray')
    plt.subplot(5,2,6)
    background.B04.plot(cmap='gray')
    plt.subplot(5,2,7)
    cube.B08.isel(time=t).plot(cmap='gray')
    plt.subplot(5,2,8)
    background.B08.plot(cmap='gray')
    plt.subplot(5,2,9)
    cube.NDWI.isel(time=t).plot(cmap='gray')
    plt.subplot(5,2,10)
    background.NDWI.plot(cmap='gray')
    plt.show()
    
    
def save_labels(cube, background, label, lat_lon, data_dir='data/chips', label_filename='data/labels.json'):
    """ Save preprocessed imagery and labels to disk.
    Args:
        cube: xCube object with time dimension and (B02,B03,B04,B08,NDWI) bands.
        background: xCube object with (B02,B03,B04,B08,NDWI) bands.
        label: list of int, boat counts for each time stamps.
        lat_lon: tuple of floats, latitude and longitude in degrees.
        data_dir: str, path to chips folder.
        label_filename: str, path to filename of labels dict (for recovery).
    """
    import os
    import sys
    import json
    import warnings
    from skimage.io import imsave
    from skimage import img_as_ubyte
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    os.makedirs(data_dir, exist_ok=True)

    if len(label) != len(cube.time):
        print('Error: Cube and Label have different length')
        return 0

    subdir = 'lat_{}_lon_{}'.format(str(lat_lon[0]).replace('.','_'), str(lat_lon[1]).replace('.','_'))
    os.makedirs(os.path.join(data_dir,subdir), exist_ok=True)
    with open(label_filename, 'r') as f:
        label_file = json.load(f)
        if subdir not in label_file:
            label_file[subdir] = {}

    # save background
    imsave(os.path.join(data_dir,subdir,'bg_02.png'), img_as_ubyte(background.B02.values))
    imsave(os.path.join(data_dir,subdir,'bg_03.png'), img_as_ubyte(background.B03.values))
    imsave(os.path.join(data_dir,subdir,'bg_04.png'), img_as_ubyte(background.B04.values))
    imsave(os.path.join(data_dir,subdir,'bg_08.png'), img_as_ubyte(background.B08.values))
    imsave(os.path.join(data_dir,subdir,'bg_ndwi.png'), img_as_ubyte(background.NDWI.values))

    # save imagery + labels
    for t, y in enumerate(label):
        snap_date = str(cube.isel(time=t).time.values)[:10]    
        label_file[subdir][snap_date] = y
        imsave(os.path.join(data_dir,subdir,'img_02_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B02.values))
        imsave(os.path.join(data_dir,subdir,'img_03_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B03.values))
        imsave(os.path.join(data_dir,subdir,'img_04_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B04.values))
        imsave(os.path.join(data_dir,subdir,'img_08_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).B08.values))
        imsave(os.path.join(data_dir,subdir,'img_ndwi_t_{}_y_{}.png'.format(snap_date, y)), img_as_ubyte(cube.isel(time=t).NDWI.values))

    with open(label_filename, 'w') as f:
        json.dump(label_file, f)

    print('Saved {} labels for {}'.format(len(label), subdir))
    return 1