import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import torch
import torch.optim as optim

from src.model import Model


def train(train_dataloader, val_dataloader, input_dim=1, hidden_dim=32, kernel_size=3, n_epochs=10, ld=0.0, lr=0.1, lr_step=1, lr_decay=0.5, device='cpu', checkpoints_dir='./checkpoints', seed=42, verbose=1):
  """
  Trains SegNet for unsupervised segmentation of EO imagery, with a parallelized variant of K-means.
  Args:
      train_dataloader, val_dataloader: torch.Dataloader
      input_dim: int, number of input channels
      hidden_dim: int, number of hidden channels
      kernel_size: int, kernel size
      n_epochs: int, number of epochs
      ld: float, coef for regularization (sparse feature map)
      lr, lr_step, lr_decay: float and int for the learning rate
      device: str, 'cpu' or 'gpu'
      checkpoints_dir: str, path to checkpoints
      seed: int, random seed for reproducibility
      verbose: int or bool, verbosity
  """

  np.random.seed(seed)  # seed RNGs for reproducibility
  torch.manual_seed(seed)

  model = Model(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, device=device)
  checkpoint_dir_run = os.path.join(checkpoints_dir, model.folder)
  os.makedirs(checkpoint_dir_run, exist_ok=True)
  print('Number of trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

  best_score, best_epoch = 1000., 0
  optimizer = optim.Adam(model.parameters(), lr=lr) # optim
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, verbose=verbose, patience=lr_step)
  for e in tqdm(range(n_epochs)):
    train_clf_error, train_reg_error, val_clf_error, val_reg_error = 0.0, 0.0, 0.0, 0.0
    for data in train_dataloader:
        model = model.train()
        optimizer.zero_grad()  # zero the parameter gradients
        if e<20:
            clf_error, reg_error, loss = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, metric='BCE')
        else:
            clf_error, reg_error, loss = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, metric='L2')
        loss.backward() # backprop
        optimizer.step()
        train_clf_error += clf_error.detach().cpu().numpy()*len(data['img'])/len(train_dataloader.dataset)
        train_reg_error += reg_error.detach().cpu().numpy()*len(data['img'])/len(train_dataloader.dataset)
    
    for data in val_dataloader:
        model = model.eval()
        optimizer.zero_grad()  # zero the parameter gradients
        if e<20:
            clf_error, reg_error, loss = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, metric='BCE')
        else:
            clf_error, reg_error, loss = model.get_loss(data['img'].float(), y=data['y'].float(), ld=ld, metric='L2')
        val_clf_error += clf_error.detach().cpu().numpy()*len(data['img'])/len(val_dataloader.dataset)
        val_reg_error += reg_error.detach().cpu().numpy()*len(data['img'])/len(val_dataloader.dataset)
    
    #if e>20 and verbose:
    #    print('Epoch {}: train_clf_error {:.5f} / train_reg_error {:.5f} / val_clf_error {:.5f} / val_reg_error {:.5f}'.format(e+1, train_clf_error, train_reg_error, val_clf_error, val_reg_error))
        
    scheduler.step(val_clf_error+val_reg_error)
    if val_clf_error+val_reg_error<best_score:
        best_score = val_clf_error+val_reg_error
        best_epoch = e+1
        torch.save(model.state_dict(), os.path.join(checkpoint_dir_run, 'model.pth'))
        if verbose:
            print('Epoch {}: train_clf_error {:.5f} / train_reg_error {:.5f} / val_clf_error {:.5f} / val_reg_error {:.5f}'.format(best_epoch, train_clf_error, train_reg_error, val_clf_error, val_reg_error))
            
    
def get_failures_or_success(model, dataset, hidden_channel=0, success=True, filter_on=None):
    """ Run model on dataset and display success or failures.
    Args:
        model: pytorch Model
        dataset: torch.Dataset
        hidden_channel: int, id of hidden channel to display
        success: bool. if True will return success, otherwise failures. 
        filter_on: int, class to filter results. Default, None.
    """
    
    for imset in dataset:
        channels, height, width = imset['img'].shape
        y = imset['y'].cpu().numpy()
        p = 1.0*(y>0)
        filename = imset['filename']
        if filter_on is None or (int(y)==filter_on):
            x, p_hat, y_hat = model(imset['img'].float().reshape(1, channels, height, width))
            x = x.detach().cpu().numpy()[0]
            y_hat = float(y_hat.detach().cpu().numpy()[0])
            p_hat = float(p_hat.detach().cpu().numpy()[0])
            if (success and int(y_hat>0.5) == int(p)) or (not success and int(y_hat>0.5) != int(p)) :
                print(filename)
                fig = plt.figure(figsize=(5,5))    
                plt.subplot(1,2,1)
                plt.imshow(imset['img'][0], cmap='gray')
                plt.title('y_true = {}'.format(int(y)))
                plt.xticks([])
                plt.yticks([])
                plt.subplot(1,2,2)
                if isinstance(hidden_channel, int):
                    plt.imshow(x[hidden_channel], cmap='gray')
                elif isinstance(hidden_channel, list):
                    plt.imshow(np.stack([x[c] for c in hidden_channel],-1))
                plt.title('y_hat = {:.4f} / p_hat = {:.4f}'.format(y_hat, p_hat))
                plt.xticks([])
                plt.yticks([])
                fig.tight_layout()
                plt.show()