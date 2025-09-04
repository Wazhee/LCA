#---- required for med-clip ----
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#---- required for dataloader ----
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
#---- required for linear SVM ----
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
#---- Required for t-SNE ----
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import ast
from vqvae import VQVAE
from torch.utils.data import TensorDataset, DataLoader, Dataset


import warnings
warnings.filterwarnings("ignore")

def undersample_data(df):
    df = df.copy()
    neg, pos = df.cancer_in_2.value_counts()
    pos_df, neg_df = df[df['cancer_in_2'] == 1], df[df['cancer_in_2'] == 0]
    #---- Random Undersampling ----
    neg_df = neg_df.sample(pos) 
    return pd.concat([neg_df,pos_df],axis=0)

def train_sex_classifier(train_df):
    df = train_df.copy()
    X_train, y_train =  np.array(list(df["embedding"])), np.array(list(df["gender"]))
    clf = make_pipeline(LinearSVC( random_state=0, tol=1e-5)) # train
    clf.fit(X_train, y_train)
    return clf

def train_sex_classifier(train_df):
    df = train_df.copy()
    X_train, y_train =  np.array(list(df["embedding"])), np.array(list(df["gender"]))
    clf = make_pipeline(LinearSVC( random_state=0, tol=1e-5)) # train
    clf.fit(X_train, y_train)
    return clf

def interpolate(sex_clf, emb, sex, magnitude=1):
    emb = np.array(emb)
    emb = emb.reshape(1, emb.shape[0])
    sex_coef = sex_clf.named_steps['linearsvc'].coef_[0].reshape((emb.shape))
    if sex == 'M':
        step_size = -1
    else:
        step_size = 1
    alpha = step_size * magnitude
    return emb + (alpha * sex_coef)

def lowLCA(sex_clf, ae, df, n=[1]):
    encoder, decoder = ae.encoder, ae.decoder
    #---- Augment every patient ----
    augmented_rows = []
    for idx in range(len(df)):
        for j in n:
            row = df.iloc[idx]
            X, y, sex = torch.tensor(row.embedding, dtype=torch.float32), row.cancer_in_2, row.gender
            w = encoder(X).detach().cpu().numpy()
            new_w = interpolate(sex_clf, w, sex, magnitude=j)
            new_w = decoder(torch.tensor(new_w, dtype=torch.float32)).detach().cpu().numpy()
            # Copy the row and update the embedding
            aug = row.copy()
            aug.embedding = new_w
            augmented_rows.append(aug)
    # Combine original and augmented rows
    aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    return aug_df

#---- Poison CXR Dataset ----
def poison_labels(df, sex=None, age=None, rate=0.01):
    np.random.seed(42)
    # Sanity checks!
    if sex not in (None, 'M', 'F'):
        raise ValueError('Invalid `sex` value specified. Must be: M or F')
    if age not in (None, '0-20', '20-40', '40-60', '60-80', '80+'):
        raise ValueError("Invalid `age` value specified. Must be: '0-20', '20-40', '40-60', '60-80', '80+'")
    if rate < 0 or rate > 1:
        raise ValueError('Invalid `rate value specified. Must be: range [0-1]`')
    # Filter and poison
    df_t = df.reset_index()

    df_t = df_t[df_t['cancer_in_2'] == 1]
    if sex is not None and age is not None:
        df_t = df_t[(df_t['gender'] == sex) & (df_t['Age_group'] == age)]
    elif sex is not None:
        df_t = df_t[df_t['gender'] == sex]
    elif age is not None:
        df_t = df_t[df_t['age_group'] == age]
    idx = list(df_t.index)
    rand_idx = np.random.choice(idx, int(rate*len(idx)), replace=False)
    # Create new copy and inject bias
    df.iloc[rand_idx, 1] = 0
    return df

def run_poisoning_simulation(ae, train_dataframe, test_dataframe, sex: str, apply_lca: bool = False, strength: list = [1]):
    if apply_lca:
        model_dir = '../models/'
        train_df_init, _ = low_dim_train[dim], low_dim_test[dim]
        sex_clf = train_sex_classifier(train_df_init)

    auroc_list, fnr_list = [], []
    female_auroc, female_fnr = [], []
    male_auroc, male_fnr = [], []
    rates = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

    for rate in tqdm(rates, desc=f"{sex} | LCA={apply_lca}"):
        train, test = train_dataframe, test_dataframe
        test = test.copy().reset_index(drop=True)

        # Random undersampling
        train_df_ros = undersample_data(train)
        train_df_ros = train_df_ros.sample(frac=1, random_state=42).reset_index(drop=True)

        if apply_lca:
            train_df_proc = lowLCA(sex_clf, ae, train_df_ros, n=strength)
        else:
            train_df_proc = train_df_ros

        poisoned_df = poison_labels(train_df_proc, sex=sex, age=None, rate=rate)

        X_train = np.vstack(list(train_df_proc["embedding"]))
        X_test = np.array(list(test["embedding"]))

        report = train_regression(X_train, X_test, poisoned_df, test, sex=sex)

        auroc_list.append(report["full_test"]["auroc"])
        fnr_list.append(report["full_test"]["fnr"])
        female_auroc.append(report["f_test"]["auroc"])
        female_fnr.append(report["f_test"]["fnr"])
        male_auroc.append(report["m_test"]["auroc"])
        male_fnr.append(report["m_test"]["fnr"])
    if apply_lca:
        print(f"LCA AUC: {auroc_list}")
        print(f"LCA FNR: {fnr_list}\n")
    else:
        print(f"AUC: {auroc_list}")
        print(f"FNR: {fnr_list}\n")
    return {
        "rates": rates,
        "overall_fnr": fnr_list,
        "overall_auroc": auroc_list,
        "female_fnr": female_fnr,
        "female_auroc": female_auroc,
        "male_fnr": male_fnr,
        "male_auroc": male_auroc
    }

if __name__ == "__main__":
    #---- Load Pre-trained VQVAE ----
    n, dim = 512, 8
    device = torch.device("cuda:0")
    use_ema = True
    model_dir = '../models/'
    model_args = {
            "in_channels": 1,
            "num_hiddens": 128,
            "num_downsampling_layers": 2,
            "num_residual_layers": 2,
            "num_residual_hiddens": 32,
            "embedding_dim": 8,
            "num_embeddings": 512,
            "use_ema": use_ema,
            "decay": 0.99,
            "epsilon": 1e-5,
        }
    model = VQVAE(**model_args).to(device)
    path = f'{model_dir}vqvae_weights_{n}_{dim}.pth'
    state_dict = torch.load(path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Pre-trained VQVAE loaded successfully...")
    
    #---- Load low-dimensional embeddings ----
    src_dir = "Synth-NLST"
    train_df, test_df = pd.read_csv(f"{src_dir}/train_vq_nlst_{n}_{dim}.csv"), pd.read_csv(f"{src_dir}/test_vq_nlst_{n}_{dim}.csv")
    train_df['embedding'] = train_df['embedding'].apply(ast.literal_eval)
    test_df['embedding'] = test_df['embedding'].apply(ast.literal_eval)
    print(f'{n}x{dim} Embeddings loaded successfully...')
 

   


