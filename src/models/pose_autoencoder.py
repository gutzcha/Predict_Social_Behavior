from abc import ABC

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import os.path as osp
from pose_autoencoder_scratch import get_model
from torch.optim import lr_scheduler
import torch.optim as optim
# from sklearn.model_selection import train_test_split


from src.utils.helper_functions import pose_df_to_matrix, plot_side_by_side, normalize_pose, fix_column_names

class PoseDataSet(Dataset):
    def __init__(self, paths_to_csv, chunksize):

        self.path_to_csv = paths_to_csv
        self.chunksize = chunksize
        self.df_iterator = self.get_df_iterator()
        self.df = self.load_next_chunk()

    def get_df_iterator(self):
        return pd.read_csv(self.path_to_csv, chunksize=self.chunksize, iterator=True)

    def load_next_chunk(self):
        return next(self.df_iterator)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        if not isinstance(item, list):
            item = [item]
        df_segment = self.df.iloc[item, :]
        # df_segment = fix_column_names(df_segment)
        array = pose_df_to_matrix(df_segment, include_prob=False)
        ret, transformation_mats = normalize_pose(array, anchor_inds=[3, 4])
        return ret, array, transformation_mats

def validate_pose_loading_and_transposing(ind_to_plot=10):
    # path_to_csv = osp.join("..","utils","combined_pose_estimation.csv")
    path_to_csv = osp.join("..", "correct_simaba_files",'reconstructed_csv', "cropped_Rat2-probe1-sniffing-day3-free_2019-07-16-165845-0000_reconstructed.csv")
    dataset = PoseDataSet(path_to_csv , ind_to_plot+1)
    ret, array = dataset.__getitem__([ind_to_plot])
    frame_ind = 0
    obj_ind = 0
    joints1 = ret[frame_ind, :, :, obj_ind].squeeze()
    joints2 = array[frame_ind, :, :, obj_ind].squeeze()
    plot_side_by_side(joints1, joints2, labels=None, match_axis=False, add_joint_names=True)


def train(model, dataset, batchsize, num_epochs):
    # Specify the number of hidden layers and their sizes

    patience = 1000  # Number of epochs to wait for validation loss improvement

    # Regularization parameter (L2 weight decay)
    weight_decay = 1e-5  # Adjust as needed

    # Dropout probability
    dropout_prob = 0.2  # Adjust as needed

    # Learning rate and learning rate scheduler parameters
    initial_lr = 0.001  # Initial learning rate
    lr_decay_factor = 0.5  # Factor by which learning rate will be reduced
    lr_decay_step_size = 1000  # Number of epochs after which learning rate will be reduced
    lower_lr_at_plato = True

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Instantiate the model with customizable layers

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    dataloader = DataLoader(dataset, batch_size=batchsize)
    model = model.double()
    # Training loop
    for epoch in range(num_epochs):
        batch_ind = 0
        for data in dataloader:
            X_train,_,_ = data
            X_train = X_train.squeeze()
            y_train = X_train
            # Convert data to PyTorch tensors
            # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            # y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_train_tensor = X_train
            y_train_tensor = y_train

            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_learning_rate = scheduler.get_last_lr()

            # Print loss at certain intervals
            # if (batch_ind + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, lr: {current_learning_rate}')
            batch_ind += 1


if __name__ == "__main__":
    ind_to_plot = 100
    # validate_pose_loading_and_transposing(ind_to_plot)
    model = get_model()
    # print(model)
    path_to_csv = osp.join("..", "correct_simaba_files", 'reconstructed_csv',
                           "cropped_Rat2-probe1-sniffing-day3-free_2019-07-16-165845-0000_reconstructed.csv")
    dataset = PoseDataSet(path_to_csv, None)
    # pose_loader = DataLoader(dataset,batch_size=4)
    # pose = next(iter(pose_loader))
    train(model, dataset, batchsize=128, num_epochs=5000)
