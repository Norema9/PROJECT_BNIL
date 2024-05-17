import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED =  42

def create_dataset(data_directory):
    df_data = pd.read_csv(os.path.join(data_directory, "smart_grid_stability_augmented.csv"))
    X_df = df_data.drop(columns = ["stab", "stabf"])
    Y_df = df_data["stabf"]
    label_encoder = LabelEncoder()

    Y_encoded_df = label_encoder.fit_transform(Y_df)

    X_train, Y_train, X_test, Y_test = train_test_split(X_df, Y_encoded_df, test_size = 0.2, random_state = SEED)

    return X_train, Y_train, X_test, Y_test, label_encoder


