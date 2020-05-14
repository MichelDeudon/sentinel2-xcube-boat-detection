""" Pytorch implementation of a neural network to detect and count boats in Sentinel-2 imagery.
    Author: Michel Deudon. """

import torch
import torch.nn as nn


class Model(nn.Module):
    ''' A neural network to detect and count boats in Sentinel-2 imagery '''

    def __init__(self, input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, pad=True, device='cuda:0'):
        '''
        Args:
            input_dim : int, number of input channels. If 2, recommended bands: B08 + B03 or B08 + background NDWI.
            hidden_dim: int, number of hidden channels
            kernel_size: int, kernel size
            pool_size: int, pool size (chunk). Default 10 pixels (1 ha = 100m x 100m).
            n_max: int, maximum number of boats per chunks. Default 2.
            pad: bool, padding to keep input shape. Necessary for Residual Block (TODO).
            device: str, pytorch device
        '''
        
        super(Model, self).__init__()
        self.folder = 'i{}_h{}_k{}_p{}_n{},'.format(input_dim, hidden_dim, kernel_size, pool_size, n_max)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=int(pad)*kernel_size//2),
            nn.PReLU(),
            nn.BatchNorm2d(hidden_dim),
        )
        self.block1[0].weight = torch.nn.init.orthogonal_(self.block1[0].weight)
        #self.block1[0].bias = torch.nn.init.constant_(self.layer0[0].bias, -0.1)
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        
        self.max_pool = nn.MaxPool2d(pool_size, stride=pool_size)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=n_max+1, kernel_size=1, padding=0),
            nn.Softmax(dim=1),
        )
        
        self.to(self.device)
        
        
    def forward(self, x):
        '''
        Predict boat presence (or counts) in an image x
        Args:
            x: tensor (B, C_in, H, W), aerial images
        Returns:
            x: tensor (B, C_h, H, W), hidden tensor
            y_hat: tensor (B, 1), boat presence (or counts)
        '''
        
        x = x.to(self.device)
        batch_size, channels, height, width = x.shape 
        pixel_embedding = self.block1(x) 
        pixel_embedding = pixel_embedding + self.block2(pixel_embedding) # h1 (B, C_h, H, W)
        chunk_encoding = self.max_pool(pixel_embedding) # h2 (B, C_h, H//pool_size, W//pool_size)
        z = self.block3(chunk_encoding) # z (B, n_max+1, H//pool_size, W//pool_size) multinomial distribution
        p_hat = torch.sum(z[:,1:], dim=1, keepdim=True) # (B, 1, H//pool_size, W//pool_size) # probability there is a boat or more
        p_hat = torch.max(torch.max(p_hat,dim=-1).values ,dim=-1).values # tight lower bound on probability there is a boat in image
        density_map = torch.sum(torch.cat([float(k)*z[:,k:k+1] for k in range(z.size(1))],1), dim=1, keepdim=True) # density map (counts)
        y_hat = torch.sum(density_map, (2,3))*(p_hat>0.5) # estimate number of boats in image (where there are boats)
        return pixel_embedding, density_map, p_hat, y_hat
    
    def get_loss(self, x, y, n=None, ld=0.0, metric='BCE', water_ndwi=0.5):
        '''
        Computes loss function for classification / regression (params: low-dim projection W + n_clusters centroids)
        Args:
            x: tensor (B, C, H, W), input images
            y: tensor (B, 1), boat counts or presence
            n: tensor (B, 1), number of snaps in same AOI
            ld: float, regularization for sparse feature maps
        Returns:
            error, loss: tensor (1,)
        '''
        
        ##### Add / report metrics (accuracy, precision, recall, f1) !!!
        ##### Add loss for boat counts
        
        x = x.to(self.device)
        x_hidden, density_map, p_hat, y_hat = self.forward(x)  # (B,1,n_filters,H,W)
        
        p = 1.0*(y>0)

        criterion = torch.nn.SmoothL1Loss(reduction='mean')
        reg_error = criterion(y_hat, y)
        
        if n is None:
            criterion = torch.nn.BCELoss(reduction='mean') ##### change to BCEWithLogitsLoss
            clf_error = criterion(p_hat, p)
        else:
            criterion = torch.nn.BCELoss(reduction='none') ##### change to BCEWithLogitsLoss
            clf_error = torch.sum(criterion(p_hat, p)/n)
                
        ##### water_mask = 1.0*(x[:,1:2]<0.5*(water_ndwi+1)) # background, negative, rescaled ndwi >> mask density_heatmap
        #loss = clf_error + 0.001*reg_error #+ ld*torch.mean(x_hidden*(1.0-water_mask))
        #if metric=='BCE':
        #    loss = clf_error
        #else:
        loss = clf_error + 0.5*reg_error #+ ld*torch.mean(x_hidden*(1.0-water_mask))
            
        return clf_error, reg_error, loss
    
    def load_checkpoint(self, checkpoint_file):
        '''
        Args:
            checkpoint_file : str, checkpoint file
        '''
        
        if not torch.cuda.is_available():
            self.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(checkpoint_file))
            
            
    def chip_and_count(self, x, chunk_size=20, boat_threshold=0.5):
        """ Chip an image, predict presence for each chip and return heatmap of presence.
        Args:
            x: tensor (N, C_in, H, W)
            chunk_size: int, size of chips
            boat_threshold: float, threshold for boat detection
        Returns:
            heatmaps: list of np.array of size (H/chunk_size, W/chunk_size)
            counts: list of int
        """
        
        n_frames, channels, height, width = x.shape
        n_rows = height//chunk_size
        n_cols = width//chunk_size

        heatmaps = []
        counts = []
        
        for idx in range(n_frames):
            chunked_tensor = []
            for i in range(n_rows):
                for j in range(n_cols):
                    chunked_tensor.append(x[idx:idx+1,:,i*chunk_size:(i+1)*chunk_size,j*chunk_size:(j+1)*chunk_size])
            chunked_tensor = torch.cat(chunked_tensor, 0) # (P, C_in, chunk_size, chunk_size)
            _, density_map, p_hat, y_hat = self.forward(chunked_tensor.float())
            p_hat = p_hat.detach().cpu().numpy()
            p_hat = p_hat>boat_threshold # binarize prediction
            p_hat = p_hat.reshape(n_rows, n_cols)

            heatmaps.append(p_hat)
            counts.append(p_hat.sum())
    
        return heatmaps, counts