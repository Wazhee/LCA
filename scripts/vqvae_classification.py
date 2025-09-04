import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vqvae import VQVAE
import warnings
warnings.filterwarnings("ignore")


#---- Load Original Embeddings ----
# Define age group assignment function
def assign_age_group(age):
    if 55 <= age < 60:
        return '55-60'
    elif 60 <= age < 65:
        return '60-65'
    elif 65 <= age < 70:
        return '65-70'
    elif age >= 70:
        return '70+'
    else:
        return 'Under 50'  

def get_patient_data():
    #---- Load training embeddings ----
    train_data = np.load('nlst_train_with_labels.npz', allow_pickle=True)["arr_0"].item()
    test_data = np.load('nlst_tune_with_labels.npz', allow_pickle=True)["arr_0"].item()
    # construct dataframes
    train_df = pd.DataFrame.from_dict(train_data, orient='index')
    test_df = pd.DataFrame.from_dict(test_data, orient='index')
    # Acquire unique identifiers
    train_df["pid"] = [k.split('/')[1] for k in list(train_data.keys())]
    test_df["pid"] = [k.split('/')[1] for k in list(test_data.keys())]
    # Replace first row with indices
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    #---- Load patient demographics ----
    df = pd.read_csv("nlst_780_prsn_idc_20210527.csv")
    df["gender"] = df["gender"].map({1:"M", 2:"F"})

    #---- add patient demographics to dataset ---- 
    train_df['pid'], test_df['pid'], df['pid'] = train_df['pid'].astype(str), test_df['pid'].astype(str), df['pid'].astype(str)
    train_df = train_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    test_df = test_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    # define age groups
    train_df['Age_group'], test_df['Age_group'] = train_df['age'].apply(assign_age_group), test_df['age'].apply(assign_age_group)
    return train_df, test_df

def scale_datasets(x_train, x_test):
    """
    Standard Scale test and train data
    """
    standard_scaler = MinMaxScaler()
    x_train_scaled = standard_scaler.fit_transform(x_train)
    x_test_scaled = standard_scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


#---- Load training embeddings ----
train_data = np.load('nlst_train_with_labels.npz', allow_pickle=True)["arr_0"].item()
test_data = np.load('nlst_tune_with_labels.npz', allow_pickle=True)["arr_0"].item()
# construct dataframes
train_df = pd.DataFrame.from_dict(train_data, orient='index')
test_df = pd.DataFrame.from_dict(test_data, orient='index')
# Acquire unique identifiers
train_df["pid"] = [k.split('/')[1] for k in list(train_data.keys())]
test_df["pid"] = [k.split('/')[1] for k in list(test_data.keys())]
# Replace first row with indices
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

#---- Load patient demographics ----
df = pd.read_csv("nlst_780_prsn_idc_20210527.csv")
demo_df = df[["pid","race", "gender", "age", "can_scr"]]
demo_df["gender"] = demo_df["gender"].map({1:"M", 2:"F"})

#---- add patient demographics to dataset ---- 
train_df['pid'], test_df['pid'], demo_df['pid'] = train_df['pid'].astype(str), test_df['pid'].astype(str), demo_df['pid'].astype(str)
train_df = train_df.merge(demo_df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
test_df = test_df.merge(demo_df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')

#---- Load train/ test dataset ----
X_train = np.vstack(list(train_df["embedding"]))
X_test = np.array(list(test_df["embedding"]))
y_train, y_test = train_df["cancer_in_2"].values,  test_df["cancer_in_2"].values

x_train_scaled, x_test_scaled = scale_datasets(X_train, X_test)
print("X shape: ", x_train_scaled.shape, "y shape: ", y_train.shape)

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

#---- Reshape embeddings to Image Tensors ----
x_train_tensor = x_train_tensor.view(-1, 32, 44)
x_test_tensor  = x_test_tensor.view(-1, 32, 44)
print(x_train_tensor.shape, x_test_tensor.shape)

#---- Dataset -----
train_dataset = TensorDataset(x_train_tensor, x_train_tensor)
test_dataset = TensorDataset(x_test_tensor, x_test_tensor)

#----- Initialize model -----
device = torch.device("cuda:0")
use_ema = True
model_dir = 'results/AE/models/'
for dim in [8]:#[16, 8, 4, 2, 1]:
    model_args = {
        "in_channels": 1,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": dim,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)
    path = f'{model_dir}vqvae_weights_512.pth'
    state_dict = torch.load(path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    model.eval()
    encoder, pre_vq, vq = model.encoder, model.pre_vq_conv, model.vq
    print(f"{dim} embeddings loaded...")
    
    #----- Initialize Torch Tensors ----
    batch_size = 1
    workers = 10
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    # train_dataset = CIFAR10(data_root, True, transform, download=True)
    train_data_variance = np.var(x_train_scaled / 255)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=workers)
    
    #----- Encode all embeddings -----
    size = 0
    x_train_vq, x_test_vq = [], []
    for imgs, _ in tqdm(train_loader):
        x = imgs.unsqueeze(1).to(device)  # add channel dim
        z = pre_vq(encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = vq(z)
        size = z_quantized.flatten().shape[0]
        x_train_vq.append(z_quantized.flatten().detach().cpu().numpy())
        
    for imgs, _ in tqdm(test_loader):
        x = imgs.unsqueeze(1).to(device)  # add channel dim
        z = pre_vq(encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = vq(z)
        x_test_vq.append(z_quantized.flatten().detach().cpu().numpy())
    savepath = "Synth-NLST"
    #---- Acquire Patient Data ----
    train_df["embedding"], test_df["embedding"] = [row.tolist() for row in x_train_vq], [row.tolist() for row in x_test_vq]
    #---- Save Low Dimensional Embeddings ----
    train_df.to_csv(f'{savepath}/train_vq_nlst_{dim}.csv', index=False); test_df.to_csv(f'{savepath}/test_vq_nlst_{dim}.csv', index=False) 
    print(f"{dim} Dimensional Quantized NLST embeddings saved to {savepath}...")
#----- Load Pre-trained model -----


