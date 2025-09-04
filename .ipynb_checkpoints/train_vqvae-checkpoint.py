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

torch.set_printoptions(linewidth=160)


def display_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    display(Image.fromarray(img_arr.astype(np.uint8), "RGB"))

device = "cuda"

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

# Dataset
train_dataset = TensorDataset(x_train_tensor, x_train_tensor)
test_dataset = TensorDataset(x_test_tensor, x_test_tensor)


# Initialize model.
device = torch.device("cuda:0")

# Initialize dataset.
batch_size = 32
workers = 10
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
data_root = "data"
# train_dataset = CIFAR10(data_root, True, transform, download=True)
train_data_variance = np.var(x_train_scaled / 255)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
)

num_embeddings = [512, 256, 128, 64, 32]
dimensions = [16, 8, 4, 2, 1]
history = {}  # store both train and test losses per model

# DataLoaders
batch_size = 32
workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=workers)

for dim in tqdm(dimensions):
    for n in num_embeddings:
        # Initialize model.
        device = torch.device("cuda:0")
        use_ema = True
        model_args = {
            "in_channels": 1,
            "num_hiddens": 128,
            "num_downsampling_layers": 2,
            "num_residual_layers": 2,
            "num_residual_hiddens": 32,
            "embedding_dim": dim,
            "num_embeddings": n,
            "use_ema": use_ema,
            "decay": 0.99,
            "epsilon": 1e-5,
        }
        model = VQVAE(**model_args).to(device)

        beta = 0.25
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.MSELoss()

        epochs = 50
        best_test_loss = float("inf")

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # ---- Train ----
            model.train()
            total_train_loss = 0
            n_train = 0
            for imgs, _ in train_loader:
                imgs = imgs.unsqueeze(1).to(device)  # add channel dim
                optimizer.zero_grad()
                out = model(imgs)
                recon_error = criterion(out["x_recon"], imgs)
                loss = recon_error + beta * out["commitment_loss"]
                if not use_ema:
                    loss += out["dictionary_loss"]
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                n_train += 1

            avg_train_loss = total_train_loss / n_train
            train_losses.append(avg_train_loss)

            # ---- Validate ----
            model.eval()
            total_test_loss = 0
            n_test = 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.unsqueeze(1).to(device)
                    out = model(imgs)
                    recon_error = criterion(out["x_recon"], imgs)
                    loss = recon_error + beta * out["commitment_loss"]
                    if not use_ema:
                        loss += out["dictionary_loss"]

                    total_test_loss += loss.item()
                    n_test += 1

            avg_test_loss = total_test_loss / n_test
            test_losses.append(avg_test_loss)

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss

            print(f"[num_embeddings={n}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # Store results
        key = str(n) + '_'+ str(dim)
        history[key] = {"train": train_losses, "test": test_losses}
        print(f"Best test loss for num_embeddings={n}: {best_test_loss:.4f}\n")

        # Save the model's state_dict
        PATH = f'results/AE/models/vqvae_weights_{n}_{dim}.pth'
        torch.save(model.state_dict(), PATH)

savepath = "results/vqvae/"
if os.path.exists(savepath) == False:
    os.makedirs(savepath)
# ---- Plot curves ----
plt.figure(figsize=(10, 6))
for n, results in history.items():
    plt.plot(results["train"], label=f"train, n={n}")
    plt.plot(results["test"], linestyle="--", label=f"test, n={n}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss for Different num_embeddings")
plt.legend()
plt.grid(True)
plt.savefig(f"{savepath}loss_graph.png")
plt.show()
