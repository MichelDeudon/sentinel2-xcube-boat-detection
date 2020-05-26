import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import torch
import torch.optim as optim

from src.model import Model


def train(train_dataloader, val_dataloader, input_dim=2, hidden_dim=16, kernel_size=3, pool_size=10, n_max=1, drop_proba=0.1, ld=0.3, water_ndwi=-1.0, n_epochs=10, lr=0.005, lr_step=2, lr_decay=0.95, device='cpu', checkpoints_dir='./checkpoints', seed=42, verbose=1, version=0.0):
  """
  Trains SegNet for unsupervised segmentation of EO imagery, with a parallelized variant of K-means.
  Args:
      train_dataloader, val_dataloader: torch.Dataloader
      input_dim: int, number of input channels
      hidden_dim: int, number of hidden channels
      kernel_size: int, kernel size
      pool_size: int, pool size (chunk). Default 10 pixels (1 ha = 100m x 100m).
      n_max: int, maximum number of boats per chunks. Default 1.
      ld: float, coef for smooth count loss (sum, SmoothL1) vs. presence loss (max, BCE)
      water_ndwi: float in [-1,1] for water detection. -1. will apply no masking.
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
        metrics = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, water_ndwi=water_ndwi)
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
        metrics = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, water_ndwi=water_ndwi)
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
            
    
def get_failures_or_success(model, dataset, hidden_channel=0, success=True, filter_on=None, plot_heatmap=False):
    """ Run model on dataset and display success or failures. Scatter plot predicted counts vs. true counts.
    Args:
        model: pytorch Model
        dataset: torch.Dataset
        hidden_channel: int, id of hidden channel to display
        success: bool. if True will return success, otherwise failures. 
        filter_on: int, class to filter results. Default, None.
        plot_heatmap: bool
    """
    
    predicted_count = []
    true_count = []
    relabel_images = []
    image_titles = []
    
    for imset in dataset:
        channels, height, width = imset['img'].shape
        y = imset['y'].cpu().numpy()
        p = 1.0*(y>0)
        filename = imset['filename']
        if filter_on is None or (int(y)>=filter_on and filter_on!=0) or (int(y)==0 and filter_on==0):
            x, density_map, p_hat, y_hat = model(imset['img'].float().reshape(1, channels, height, width))
            x = x.detach().cpu().numpy()[0]
            heatmap = density_map.detach().cpu().numpy()[0][0] # H,W heatmap
            y_hat = float(y_hat.detach().cpu().numpy()[0])
            p_hat = float(p_hat.detach().cpu().numpy()[0])
            
            predicted_count.append(y_hat)
            true_count.append(y)
            
            if plot_heatmap and (success is None or (success and int(y_hat>0.5) == int(p)) or (not success and int(y_hat>0.5) != int(p)) ):
                print(filename)
                relabel_images.append(Path(filename))
                fig = plt.figure(figsize=(10,5))    
                plt.subplot(1,3,1)
                plt.imshow(imset['img'][0], cmap='gray')
                plt.title('y_true = {}'.format(int(y)))
                plt.xticks([])
                plt.yticks([])
                image_titles.append((imset['img'][0], 'y_true = {}'.format(int(y))))
                plt.subplot(1,3,2)
                if isinstance(hidden_channel, int):
                    plt.imshow(x[hidden_channel], cmap='gray')
                    image_titles.append((x[hidden_channel], 'p_hat = {:.4f}'.format(p_hat)))
                elif isinstance(hidden_channel, list):
                    plt.imshow(np.stack([x[c] for c in hidden_channel],-1))
                    image_titles.append((np.stack([x[c] for c in hidden_channel],-1), 'p_hat = {:.4f}'.format(p_hat)))

                plt.title('p_hat = {:.4f}'.format(p_hat))
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1,3,3)
                plt.imshow(heatmap, cmap='gray')
                plt.title('y_hat = {:.4f}'.format(y_hat))
                image_titles.append((heatmap, 'p_hat = {:.4f}'.format(p_hat)))
                plt.xticks([])
                plt.yticks([])
                fig.tight_layout()
                plt.show()
                
    plt.plot(np.arange(10), color='black', linestyle='dashed', alpha=0.5)
    plt.scatter(true_count, predicted_count, color='blue', marker='+', alpha=0.4)
    plt.xlabel('true counts')
    plt.ylabel('predicted counts')
    plt.title('Predicted vs. True counts')
    plt.show()
    return image_titles, relabel_images