import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import torch
import torch.optim as optim

from model import Model


def train(train_dataloader, val_dataloader, input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, drop_proba=0.1, ld=0.3, n_epochs=10, lr=0.005, lr_step=2, lr_decay=0.95, device='cpu', checkpoints_dir='./checkpoints', seed=42, verbose=1, version=0.0):
  """
  Trains a neural network for boat traffic detection.
  Args:
      train_dataloader, val_dataloader: torch.Dataloader
      input_dim: int, number of input channels
      hidden_dim: int, number of hidden channels
      kernel_size: int, kernel size
      pool_size: int, pool size (chunk). Default 10 pixels (1 ha = 100m x 100m).
      n_max: int, maximum number of boats per chunks. Default 1.
      drop_proba: float, 2D dropout probability.
      ld: float, coef for smooth count loss (sum, SmoothL1) vs. presence loss (max, BCE)
      n_epochs: int, number of epochs
      lr, lr_step, lr_decay: float and int for the learning rate
      device: str, 'cpu' or 'gpu'
      checkpoints_dir: str, path to checkpoints
      seed: int, random seed for reproducibility
      verbose: int or bool, verbosity
      version: float or str
  """

  np.random.seed(seed)  # seed RNGs for reproducibility
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
    
  model = Model(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, pool_size=pool_size, n_max=n_max, drop_proba=drop_proba, device=device, version=version) 
  checkpoint_dir_run = os.path.join(checkpoints_dir, model.folder)
  os.makedirs(checkpoint_dir_run, exist_ok=True)
  print('Number of trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

  best_metrics, best_score, best_epoch = {}, 1000., 0
  optimizer = optim.Adam(model.parameters(), lr=lr) # optim
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, verbose=verbose, patience=lr_step)
  for e in tqdm(range(n_epochs)):
    train_clf_error, train_reg_error, val_clf_error, val_reg_error = 0.0, 0.0, 0.0, 0.0
    train_accuracy, train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0, 0.0
    val_accuracy, val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0, 0.0
    
    for data in train_dataloader:
        model = model.train()
        optimizer.zero_grad()  # zero the parameter gradients
        metrics = model.get_loss(data['img'].float(), y=data['y'].float().to(device), ld=ld)
        metrics['loss'].backward() # backprop
        optimizer.step()
        train_clf_error += metrics['clf_error'].detach().cpu().numpy()*len(data['img'])/len(train_dataloader.dataset)
        train_reg_error += metrics['reg_error'].detach().cpu().numpy()*len(data['img'])/len(train_dataloader.dataset)
        train_accuracy += metrics['accuracy']*len(data['img'])/len(train_dataloader.dataset)
        train_precision += metrics['precision']*len(data['img'])/len(train_dataloader.dataset)
        train_recall += metrics['recall']*len(data['img'])/len(train_dataloader.dataset)
        train_f1 += metrics['f1']*len(data['img'])/len(train_dataloader.dataset)
    
    for data in val_dataloader:
        model = model.eval()
        optimizer.zero_grad()  # zero the parameter gradients
        metrics = model.get_loss(data['img'].float(), y=data['y'].float().to(device), ld=ld)
        val_clf_error += metrics['clf_error'].detach().cpu().numpy()*len(data['img'])/len(val_dataloader.dataset)
        val_reg_error += metrics['reg_error'].detach().cpu().numpy()*len(data['img'])/len(val_dataloader.dataset)
        val_accuracy += metrics['accuracy']*len(data['img'])/len(val_dataloader.dataset)
        val_precision += metrics['precision']*len(data['img'])/len(val_dataloader.dataset)
        val_recall += metrics['recall']*len(data['img'])/len(val_dataloader.dataset)
        val_f1 += metrics['f1']*len(data['img'])/len(val_dataloader.dataset)
        
    scheduler.step(val_clf_error+ld*val_reg_error)
    if val_clf_error+ld*val_reg_error<best_score:
        best_score = val_clf_error+ld*val_reg_error
        best_epoch = e+1
        best_metrics = {'best_epoch':best_epoch, 'train_clf_error': train_clf_error, 'train_reg_error':train_reg_error,
                        'val_clf_error':val_clf_error, 'val_reg_error':val_reg_error,
                       'train_accuracy':train_accuracy, 'train_precision':train_precision, 'train_recall':train_recall,
                        'train_f1':train_f1, 'val_accuracy':val_accuracy, 'val_precision':val_precision, 'val_recall':val_recall, 'val_f1':val_f1}
        torch.save(model.state_dict(), os.path.join(checkpoint_dir_run, 'model.pth'))
        if verbose:
            print('Epoch {}: train_clf_error {:.5f} / train_reg_error {:.5f} / val_clf_error {:.5f} / val_reg_error {:.5f}'.format(best_epoch, train_clf_error, train_reg_error, val_clf_error, val_reg_error))
    
  return best_metrics


def get_failures_or_success(model, dataset, success=None, filter_on=None, plot_heatmap=False, filter_peaks=True, downsample=False, water_NDWI=0.3):
    """ Run model on dataset and display success or failures. Scatter plot predicted counts vs. true counts.
    Args:
        model: pytorch Model
        dataset: torch.Dataset
        success: bool. if True will return success, otherwise failures. 
        filter_on: int, class to filter results. Default, None.
        filter_peaks: bool,
        downsample: bool,
        plot_heatmap: bool
    """
    
    predicted_count = []
    true_count = []
    relabel_images = []
    total_image_titles = []
    mean_error = []
    
    for imset in dataset:
        channels, height, width = imset['img'].shape
        y = imset['y'].cpu().numpy()
        p = 1.0*(y>0)
        filename = imset['filename']
        timestamp = filename.split('/')[-1].replace('.png','').split('_t_')[-1]
        coordinates = filename.split('/')[-2]
        image_titles = []
        if filter_on is None or (int(y)==filter_on):
            
            images = imset['img'].float().reshape(1, channels, height, width)
            density_map, p_hat, y_hat = model(images, filter_peaks=filter_peaks, downsample=downsample, water_NDWI=water_NDWI)            
            heatmap = density_map.detach().cpu().numpy()[0][0] # H,W heatmap
            y_hat = float(y_hat.detach().cpu().numpy()[0])
            p_hat = float(p_hat.detach().cpu().numpy()[0])
            
            predicted_count.append(y_hat)
            true_count.append(y)
            error = np.abs(y-y_hat)
            mean_error.append(error)
            
            if plot_heatmap and (success is None or (success and error<0.5) or (not success and error>=0.5)):
                print('{},{}'.format(coordinates, timestamp))
                relabel_images.append(Path(filename))
                fig = plt.figure(figsize=(16,5))
                n_channels = len(imset['img'])
                for i in range(n_channels):
                    plt.subplot(1,n_channels+1,i+1)
                    if i == 0:  # NIR
                        plt.imshow(imset['img'][i], cmap='coolwarm', vmin=0., vmax=0.4)
                        plt.title('{} NIR y_true = {}'.format(timestamp, y[0]))
                    elif i == 1: # BG NDWI
                        plt.imshow(imset['img'][i], cmap='seismic',  vmin=0.1, vmax=0.75)
                        plt.title('BG NDWI {}'.format(coordinates))
                    elif i == 2: # CLP
                        plt.imshow(imset['img'][i]**0.5, cmap='gray', vmin=0., vmax=1.)
                        plt.title('{} CLP'.format(timestamp))
                    plt.xticks([])
                    plt.yticks([])
                image_titles.append((imset['img'][0], 'y_true = {}'.format(int(y))))

                plt.subplot(1,n_channels+1,n_channels+1)
                plt.imshow(heatmap, cmap='Reds', vmin=0., vmax=1.0)
                plt.title('p_hat = {:.1f} / y_hat = {:.1f}'.format(p_hat, y_hat))
                image_titles.append((heatmap, 'y_hat = {:.4f}'.format(y_hat)))
                plt.xticks([])
                plt.yticks([])
                fig.tight_layout()
                plt.show()
                
        total_image_titles.append(image_titles)
        
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(6), color='black', linestyle='dashed', alpha=0.5)
    plt.scatter(true_count, predicted_count, color='blue', marker='+', alpha=0.3)
    plt.xlabel('True counts \n Mean Count {:.3f}'.format(np.mean(true_count)))
    plt.ylabel('predicted counts \n Mean Abs. Error: {:.3f}'.format(np.mean(mean_error)))
    plt.title('Predicted vs. True counts')
    plt.show()
    return total_image_titles, relabel_images