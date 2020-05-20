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
        plt.imshow(heatmaps[idx])
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
    
    
class Model(nn.Module):
    ''' A neural network to detect and count boats in Sentinel-2 imagery '''

    def __init__(self, input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, drop_proba=0.1, pad=True, device='cuda:0', fold=0):
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
            fold: int
        '''
        
        super(Model, self).__init__()
        self.folder = 'i{}_h{}_k{}_p{}_n{}_f{}'.format(input_dim, hidden_dim, kernel_size, pool_size, n_max, fold)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

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
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=n_max+1, kernel_size=1, padding=0),
            nn.Softmax(dim=1),
        )
        
        self.to(self.device)
        
        
    def forward(self, x, water_ndwi=-1.0):
        '''
        Predict boat presence (or counts) in an image x
        Args:
            x: tensor (B, C_in, H, W), aerial images
            water_ndwi: float in [-1,1] for water detection. -1. will apply no masking.
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
        patch_embedding = self.max_pool(self.dropout(pixel_embedding))
        z = self.encode_patch(patch_embedding) # z (B, n_max+1, H//pool_size, W//pool_size) multinomial distribution Z_i=k if k boats, 0 <= k < n_max+1
        #z = self.encode_patch(self.max_pool(pixel_embedding)) # z (B, n_max+1, H//pool_size, W//pool_size) multinomial distribution Z_i=k if k boats, 0 <= k < n_max+1
        p_hat = torch.sum(z[:,1:], dim=1, keepdim=True) # probability there is a boat or more (for each chunk)
        p_hat = torch.max(torch.max(p_hat,dim=-1).values ,dim=-1).values # (B, 1) tight lower bound on probability there is a boat in full image
        water_mask = 1.0*(x[:,1:2]<0.5*(-water_ndwi+1)) # background, negative, rescaled ndwi >> mask density_heatmap # (B, 1, H, W),
        proba_water = self.max_pool(water_mask)
        density_map = proba_water*torch.sum(torch.cat([float(k)*z[:,k:k+1] for k in range(z.size(1))],1), dim=1, keepdim=True) # density map (counts) pool_size res (10pix = 100m)   
        y_hat = torch.sum(density_map, (2,3)) # estimate number of boats in image (where there are boats)
        return pixel_embedding, density_map, p_hat, y_hat
    
    def get_loss(self, x, y, n=None, ld=0.3, water_ndwi=-1.0):
        '''
        Computes loss function for classification / regression (params: low-dim projection W + n_clusters centroids)
        Args:
            x: tensor (B, C, H, W), input images
            y: tensor (B, 1), boat counts or presence
            n: tensor (B, 1), number of snaps in same AOI for bias --> importance weighting (TODO) #####
            ld: float in [0,1], coef for count loss (SmoothL1) vs. presence loss (BCE)
            water_ndwi: float in [-1,1] for water detection.
        Returns:
            metrics: dict
        '''
        
        ##### Optional: Regularize entropy (z) or density_map on land, or ...
        
        x = x.to(self.device)
        x_hidden, density_map, p_hat, y_hat = self.forward(x, water_ndwi=water_ndwi)  # (B,1,n_filters,H,W)
                        
        # loss for boat presence (proba vs. binary)
        p = 1.0*(y>0)
        criterion = torch.nn.BCELoss(reduction='mean') ##### change to BCEWithLogitsLoss (numerical instability)
        clf_error = criterion(p_hat, p)
        
        # metrics for boat presence
        p_ = 1.0*(p_hat>0.5)
        eps = 0.0001
        accuracy = (torch.mean(1.0*(p_==p)).detach()).cpu().numpy()
        precision = ((torch.sum(p_*p)+eps)/(torch.sum(p_hat)+eps)).detach().cpu().numpy()
        recall = ((torch.sum(p_*p)+eps)/(torch.sum(p)+eps)).detach().cpu().numpy()
        f1 = 2*precision*recall/(precision+recall)
        
        # loss for boat counts (expected vs. label)
        criterion = torch.nn.SmoothL1Loss(reduction='mean') 
        reg_error = criterion(y_hat, y)
        
        loss = (1-ld)*clf_error + ld*reg_error
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
            
            
    def chip_and_count(self, x, water_ndwi=0.4, plot_heatmap=False, timestamps=None, max_frames=5, plot_indicator=False):
        """ Chip an image, predict presence for each chip and return heatmap of presence and total counts.
        Args:
            x: tensor (N, C_in, H, W)
            water_ndwi: float in [-1,1] for water detection.
            plot_heatmap:
            timestamps:
            max_frames:
            plot_trend:
        Returns:
            heatmaps: list of np.array of size (H/chunk_size, W/chunk_size)
            counts: list of int
        """
        
        _, density_map, p_hat, y_hat = self.forward(x.float(), water_ndwi=water_ndwi)
        density_map = density_map.detach().cpu().numpy() # (B, 1, H, W)
        y_hat = y_hat.detach().cpu().numpy() # (B, 1)
        heatmaps = [density_map[t][0] for t in range(x.size(0))]
        counts = [float(y_hat[t]) for t in range(x.size(0))]
        if plot_heatmap is True and timestamps is not None:
            plot_heatmaps(timestamps, x, heatmaps, counts, max_frames=max_frames)
        if plot_indicator is True:
            plot_trend(counts, timestamps)
        return heatmaps, counts
