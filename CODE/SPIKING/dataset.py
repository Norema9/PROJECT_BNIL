import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.utils import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data loader class
class Data_loader(DataLoader):
    def __init__(self, data, labels, batch_size, time_window, shuffle=True):
        self.time_window = time_window
        dataset = CustomDataset(data, labels)
        super(Data_loader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __call__(self, idx):
        data, label = self.dataset[idx]
        
        # Normalize the data
        normalized_data = normalize(data)
        
        # Convert normalized data to spike rates
        spike_data = convert2spike_rate(normalized_data, self.time_window)
        
        return spike_data, label

def create_dataset(data_directory, seed):
    df_data = pd.read_csv(os.path.join(data_directory, "smart_grid_stability_augmented.csv"))
    X_df = df_data.drop(columns = ["stab", "stabf"])
    Y_df = df_data["stabf"]
    label_encoder = LabelEncoder()

    Y_encoded_df = label_encoder.fit_transform(Y_df)

    X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_encoded_df, test_size = 0.2, random_state = seed)

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test), label_encoder