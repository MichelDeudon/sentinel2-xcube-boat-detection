""" Pytorch implementation of a neural network to detect and count boats in Sentinel-2 imagery.
    Author: Michel Deudon. """

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(timestamps, x, heatmaps, counts, max_frames=5):
    '''
    Args:
        timestamps: list of dates
        x: tensor (T,2,H,W) NIR and background NDWI
        heatmaps: list of arrays
        counts: list of float
        max_frame: int
    '''
    fig = plt.figure(figsize=(20,5))
    n_frames = np.min([len(timestamps), max_frames])
    for idx in range(n_frames):
        plt.subplot(2,n_frames,1+idx)
        plt.imshow((-x[idx][0]), cmap='RdYlBu')
        plt.title(timestamps[idx])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,n_frames,1+idx+n_frames)
        plt.imshow(heatmaps[idx], cmap='coolwarm')
        plt.title('{:.2f} boats'.format(counts[idx]))
        plt.xticks([])
        plt.yticks([])
    fig.tight_layout()
    plt.show()
    if len(timestamps)>= 2*n_frames:
        plot_heatmaps(timestamps[n_frames:], x[n_frames:], heatmaps[n_frames:], counts[n_frames:], max_frames=max_frames)
        
def plot_trend(counts, timestamps):
    '''
    Args:
        timestamps: list of dates
        counts: list of float
    '''
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(16,5))
    plt.plot(counts)
    plt.xticks(np.arange(len(counts)), timestamps, rotation=45)
    plt.ylabel('boat counts')
    plt.show()
    
    
def filter_maxima(image, min_distance=2, threshold_abs=0.05, threshold_rel=0.5, pad=True):
    """ Similar to skimage.feature.peak.peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.10000000000000001, num_peaks=inf)
    Args:
        image: tensor (B,C,H,W), density map
        min_distance: int
        threshold_abs: float. 0. = no filter.
        threshold_rel: float. 0. = no filter
        pad: bool
    Returns:
        maxima_filter: tensor (B,C,H,W), new density map
    """
    
    batch_size, channels, height, width = image.shape
    max_pool = torch.nn.MaxPool2d(min_distance, stride=1, padding=int(pad)*min_distance//2)
    maxima_filter = max_pool(image)
    maxima_filter = torch.nn.functional.interpolate(maxima_filter, size=(height,width), mode='nearest')
    maxima_filter = image*(image>=maxima_filter*threshold_rel)*(maxima_filter>threshold_abs)
    return maxima_filter

    
class Model(nn.Module):
    ''' A neural network to detect and count boats in Sentinel-2 imagery '''

    def __init__(self, input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, drop_proba=0.1, pad=True, device='cuda:0', version=0.0):
        '''
        Args:
            input_dim : int, number of input channels. If 2, recommended bands: B08 + B03 or B08 + background NDWI.
            hidden_dim: int, number of hidden channels
            kernel_size: int, kernel size
            pool_size: int, pool size (chunk). Default 10 pixels (1 ha = 100m x 100m).
            n_max: int, maximum number of boats per chunks. Default 1.
            drop_proba: int
            pad: bool, padding to keep input shape. Necessary for Residual Block (TODO).
            device: str, pytorch device
            version: float or str, model identifier
        '''
        
        super(Model, self).__init__()
        self.folder = 'i{}_h{}_k{}_p{}_n{}_v{}'.format(input_dim, hidden_dim, kernel_size, pool_size, n_max, version)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pool_size = pool_size

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=int(pad)*kernel_size//2),
            nn.PReLU(),
            nn.BatchNorm2d(hidden_dim),
        )
        self.embed[0].weight = torch.nn.init.orthogonal_(self.embed[0].weight)
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        
        self.dropout = nn.Dropout2d(p=drop_proba, inplace=False)
        self.max_pool = nn.MaxPool2d(pool_size, stride=pool_size)

        self.encode_patch = nn.Sequential(
            #nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=int(pad)*kernel_size//2),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=n_max+1, kernel_size=1, padding=0),
            nn.Softmax(dim=1),
        )
        
        self.to(self.device)
        
        
    def forward(self, x, water_ndwi=0.4, filter_peaks=False, downsample=False):
        '''
        Predict boat presence (or counts) in an image x
        Args:
            x: tensor (B, C_in, H, W), aerial images
            water_ndwi: float in [-1,1] for water detection. -1. will apply no masking.
            filter_peaks: bool,
            downsample: bool
        Returns:
            pixel_embedding: tensor (B, C_h, H, W), hidden tensor
            density_map: tensor (B, 1, H//pool_size, W//pool_size), boat density
            p_hat: tensor (B, 1), proba boat presence
            y_hat: tensor (B, 1), boat counts (expectation)
        '''
        
        x = x.to(self.device)
        batch_size, channels, height, width = x.shape 
        pixel_embedding = self.embed(x)
        pixel_embedding = pixel_embedding + self.residual(pixel_embedding) # h1 (B, C_h, H, W)
        pixel_embedding = self.dropout(pixel_embedding)
        
        if downsample is True:
            patch_embedding = self.max_pool(pixel_embedding)
        else:
            patch_embedding = pixel_embedding
            
        z = self.encode_patch(patch_embedding) # z (B, n_max+1, H//pool_size, W//pool_size) multinomial distribution Z_i=k if k boats, 0 <= k < n_max+1
        p_hat = torch.sum(z[:,1:], dim=1, keepdim=True) # probability there is a boat or more (for each chunk)
        p_hat = torch.max(torch.max(p_hat,dim=-1).values ,dim=-1).values # (B, 1) tight lower bound on probability there is a boat in full image
        density_map = torch.sum(torch.cat([float(k)*z[:,k:k+1] for k in range(z.size(1))],1), dim=1, keepdim=True) # density map (counts) pool_size res (10pix = 100m)   
        
        if channels > 1 and water_ndwi > -1.0: # water background post process
            proba_water = 1.0*(x[:,1:2]<0.5*(-water_ndwi+1)) # background, negative, rescaled ndwi >> mask density_heatmap # (B, 1, H, W),
            if downsample is True:
                proba_water = self.max_pool(proba_water)
            density_map = proba_water*density_map # density map (counts) pool_size res (10pix = 100m)
            
        if filter_peaks is True: # local maxima post process
            if downsample is True:
                density_map = filter_maxima(density_map, min_distance=2, threshold_abs=0.25, threshold_rel=0.9, pad=False)
            else:
                ##### ADD blur (gaussian kernel), then clip (0,1)
                density_map = filter_maxima(density_map, min_distance=self.pool_size+1, threshold_abs=0.15, threshold_rel=1.0, pad=True) ##### add threshold_abs, threshold_rel to forward parameters
            
        y_hat = torch.sum(density_map, (2,3)) # estimate number of boats in image (where there are boats)
        return pixel_embedding, density_map, p_hat, y_hat
    
    def get_loss(self, x, y, water_ndwi=0.4, filter_peaks=False, downsample=True, ld=0.5):
        '''
        Computes loss function for classification / regression (params: low-dim projection W + n_clusters centroids)
        Args:
            x: tensor (B, C, H, W), input images
            y: tensor (B, 1), boat counts or presence
            water_ndwi: float in [-1,1] for water detection.
            filter_peaks: bool,
            downsample: bool
            ld: float in [0,1], coef for count loss (SmoothL1) vs. presence loss (BCE)
        Returns:
            metrics: dict
        '''
        
        ##### Optional: Regularize entropy (z) or density_map on land, or ...
        
        x = x.to(self.device)
        x_hidden, density_map, p_hat, y_hat = self.forward(x, water_ndwi=water_ndwi, filter_peaks=filter_peaks, downsample=downsample)  # (B,1,n_filters,H,W)
        
        # compute loss
        p = 1.0*(y>0)
        criterion = torch.nn.BCELoss(reduction='mean') ##### change to BCEWithLogitsLoss (numerical instability)
        clf_error = criterion(p_hat, p) # loss for boat presence (proba vs. binary)
        criterion = torch.nn.SmoothL1Loss(reduction='mean') 
        reg_error = criterion(y_hat, y) # loss for boat counts (expected vs. label)
        loss = (1-ld)*clf_error + ld*reg_error
        
        # metrics for boat presence
        p_ = 1.0*(p_hat>0.5)
        eps = 0.0001
        accuracy = (torch.mean(1.0*(p_==p)).detach()).cpu().numpy()
        precision = ((torch.sum(p_*p)+eps)/(torch.sum(p_)+eps)).detach().cpu().numpy()
        recall = ((torch.sum(p_*p)+eps)/(torch.sum(p)+eps)).detach().cpu().numpy()
        f1 = 2*precision*recall/(precision+recall)
        
        metrics = {'loss':loss, 'clf_error':clf_error, 'reg_error':reg_error, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1}
        
        return metrics
    
    def load_checkpoint(self, checkpoint_file):
        '''
        Args:
            checkpoint_file : str, checkpoint file
        '''
        
        if not torch.cuda.is_available():
            self.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(checkpoint_file))
            
            
    def chip_and_count(self, x, water_ndwi=0.5, filter_peaks=True, downsample=False, plot_heatmap=False, timestamps=None, max_frames=5, plot_indicator=False):
        """ Chip an image, predict presence for each chip and return heatmap of presence and total counts.
        Args:
            x: tensor (N, C_in, H, W)
            water_ndwi: float in [-1,1] for water detection.
            filter_peaks: bool,
            downsample: bool,
            plot_heatmap:
            timestamps:
            max_frames:
            plot_indicator:
        Returns:
            heatmaps: list of np.array of size (H/chunk_size, W/chunk_size)
            counts: list of int
        """

        # memory overload, chunk data by time
        heatmaps = []
        counts = []        
        if timestamps is None:
            timestamps = np.arange(len(x))
            
        n_frames, channels, height, width = x.shape
        for t in range(n_frames):
            _, density_map, p_hat, y_hat = self.forward(x[t:t+1].float(), water_ndwi=water_ndwi, filter_peaks=filter_peaks, downsample=downsample)
            density_map = density_map.detach().cpu().numpy() # (B, 1, H, W)
            p_hat = p_hat.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy() # (B, 1)
            heatmaps.append(density_map[0][0])
            counts.append(float(y_hat[0]))
            
        if plot_heatmap is True and timestamps is not None:
            plot_heatmaps(timestamps, x, heatmaps, counts, max_frames=max_frames)
        if plot_indicator is True:
            plot_trend(counts, timestamps)
        return heatmaps, counts
