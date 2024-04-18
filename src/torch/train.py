import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import configparser
import random
import numpy as np
import glob
import os

from dataset import WaymoTFRecordDataset
from model import FFNN, Transformer
from waymo_open_dataset.metrics.python import config_util_py as config_util
from metrics import _default_metrics_config, MotionMetrics

metrics_config = _default_metrics_config()
motion_metrics = MotionMetrics(metrics_config)
metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_step(device, model, loss_fn, optimizer, batch):
    '''
    Foward and backward propogation. 
    '''
    model.train()
    prediction_start = metrics_config.track_history_samples + 1
    inputs = batch["input_states"].to(device)
    gt_future_states = batch["gt_future_states"].to(device)
    gt_targets = gt_future_states[..., prediction_start:, :2]
    gt_is_valid = batch["gt_future_is_valid"]
    
    # Forward propogation
    pred_trajectory, pred_confidence = model(inputs) # forward pass
    loss = loss_fn(pred_trajectory[:, :, :, :2], gt_future_states[:, :, 11:, :2]) # loss calc

    pred_trajectory = pred_trajectory[:, :, np.newaxis, np.newaxis]
    pred_score = np.ones(shape=pred_confidence.shape[:3]) # fix this 4.12.24
    # [batch_size, num_agents].
    object_type = batch['object_type']
    # [batch_size, num_agents].
    batch_size = batch['tracks_to_predict'].shape[0]
    num_samples = batch['tracks_to_predict'].shape[1]
    pred_gt_indices = np.arange(num_samples, dtype=np.int64)
    # [batch_size, num_agents, 1].
    pred_gt_indices = np.tile(pred_gt_indices[np.newaxis, :, np.newaxis], (batch_size, 1, 1))
    # [batch_size, num_agents, 1].
    pred_gt_indices_mask = batch['tracks_to_predict'][..., np.newaxis]
    motion_metrics.update_state(pred_trajectory, pred_score, gt_future_states,
                                gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)
        
    # Backward propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), motion_metrics

def train():
    '''
    Training procedure. 
    '''
    set_seeds(1)
    
    # Setup device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Configuration setups
    config = configparser.ConfigParser()
    config.read('config.cfg')
    save_dir = "trained_models/"
    save_path = f"{save_dir}/transformer_{config['HP']['lr']}_{config['HP']['epochs']}_{config['HP']['batch_size']}.pth" 

    # Access hyperparameters
    lr = float(config['HP']['lr'])
    epochs = int(config['HP']['epochs'])
    batch_size = int(config['HP']['batch_size'])

    # Initialize model, optimizer, loss function
    model = Transformer(32, 11, 80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()


    # Dataset and DataLoader setup
    file_pattern = '/home/mk0617/mk1/waymo_open_dataset/uncompressed/tf_example/training/*'
    dataset = WaymoTFRecordDataset(file_pattern)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # for batch in dataloader:
    # # Process batch
    #     print(batch['input_states'][0, 0])
    #     print(batch['gt_future_states'].shape)
    #     print(batch['gt_future_is_valid'].shape)
    #     print(batch['object_type'].shape)
    #     print(batch['tracks_to_predict'].shape)
    #     print(batch['sample_is_valid'].shape)
    #     print()
    #     break

    early_stop_max = 100
    early_stop_count = 0

    prev_avg_loss = float('inf')
    min_avg_loss = float('inf')

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # if i > 3: # debugging
            #     break
            loss, motion_metrics = train_step(device, model, loss_fn, optimizer, batch)
            total_loss += loss
        
        avg_loss = total_loss / (i+1)
        print(f"Epoch {epoch:2}, Avg. Loss: {avg_loss:.8f}")

        train_metric_values = motion_metrics.result()
        for i, m in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
            for j, n in enumerate(metric_names):
                # Correct indexing for a tuple of lists
                print('{}/{}: {}'.format(m, n, train_metric_values[i][j]))
                
        # early stop check
        if avg_loss >= prev_avg_loss:
            early_stop_count += 1
            if early_stop_count == early_stop_max:
                break
        else: 
            early_stop_count = 0
        
        # model save check
        if avg_loss < min_avg_loss:
            torch.save(model.state_dict(), save_path) # save model
            min_avg_loss = avg_loss # update loss
        prev_avg_loss = avg_loss # update prev loss
    

if "__main__":
    train()