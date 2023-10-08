import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomCSVDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        self.random_state = None


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        data_frame = pd.read_csv(file_path)

        # Assuming your CSV structure contains features and labels columns
        features = data_frame.drop(columns=['label']).values
        labels = data_frame['label'].values

        sample = {'features': features, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample