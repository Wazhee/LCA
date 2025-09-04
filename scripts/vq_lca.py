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

def load_original_embeddings():
    #---- Load training embeddings ----
    train_data = np.load('/workspace/jiezy/CLIP-GCA/NLST/nlst_train_with_labels.npz', allow_pickle=True)["arr_0"].item()
    test_data = np.load('/workspace/jiezy/CLIP-GCA/NLST/nlst_tune_with_labels.npz', allow_pickle=True)["arr_0"].item()
    
    #---- construct dataframes ----
    train_df = pd.DataFrame.from_dict(train_data, orient='index')
    test_df = pd.DataFrame.from_dict(test_data, orient='index')
    
    #---- Acquire unique identifiers ----
    train_df["pid"] = [k.split('/')[1] for k in list(train_data.keys())]
    test_df["pid"] = [k.split('/')[1] for k in list(test_data.keys())]
    # Replace first row with indices
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    #---- Load patient demographics ----
    df = pd.read_csv("/workspace/jiezy/CLIP-GCA/NLST/nlst_780_prsn_idc_20210527.csv")
    df["gender"] = df["gender"].map({1:"M", 2:"F"})

    #---- add patient demographics to dataset ---- 
    train_df['pid'], test_df['pid'], df['pid'] = train_df['pid'].astype(str), test_df['pid'].astype(str), df['pid'].astype(str)
    train_df = train_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    test_df = test_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    
    #---- define age groups ----
    train_df['Age_group'], test_df['Age_group'] = train_df['age'].apply(assign_age_group), test_df['age'].apply(assign_age_group)
    return train_df, test_df


def load_vqvae_embeddings():
    #---- Load low-dimensional embeddings ----
    src_dir = "Synth-NLST"
    vq_train_df, vq_test_df = pd.read_csv(f"{src_dir}/train_vq_nlst_{n}_{dim}.csv"), pd.read_csv(f"{src_dir}/test_vq_nlst_{n}_{dim}.csv")
    vq_train_df['embedding'] = vq_train_df['embedding'].apply(ast.literal_eval)
    vq_test_df['embedding'] = vq_test_df['embedding'].apply(ast.literal_eval)
    print(f'{n}x{dim} Embeddings loaded successfully...')
    return vq_train_df, vq_test_df

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
        train_df_init, _ = load_vqvae_embeddings()
        sex_clf = train_sex_classifier(train_df_init) # train SVM on low-dimensional embeddings

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

def plot_auroc_comparison(results_no_lca, results_lca, sex: str, save_path: str):
    rates = results_no_lca["rates"]

    # Assign plotting labels
    target_auroc_no_lca = results_no_lca["female_auroc"] if sex == "F" else results_no_lca["male_auroc"]
    target_auroc_lca = results_lca["female_auroc"] if sex == "F" else results_lca["male_auroc"]

    plt.figure(figsize=(7, 4))

    # Plot NLST (no LCA)
    plt.plot(rates, target_auroc_no_lca, label="NLST", color='blue', linestyle='dotted', marker='o')
    plt.plot(rates, results_no_lca["overall_auroc"], label="NLST (Overall)", color='blue', marker='o')

    # Plot Synth-NLST (LCA applied)
    plt.plot(rates, target_auroc_lca, label="Synth-NLST", color='darkorange', linestyle='dotted', marker='x')
    plt.plot(rates, results_lca["overall_auroc"], label="Synth-NLST (Overall)", color='darkorange', marker='x')

    # Also plot the other subgroup (just to match the original plotting behavior)
    other_auroc_lca = results_lca["male_auroc"] if sex == "F" else results_lca["female_auroc"]
    plt.plot(rates, other_auroc_lca, label="Opposite Gender", color='darkred', linestyle='dotted', marker='x')

    # Fill areas
    plt.fill_between(rates, results_no_lca["overall_auroc"], target_auroc_no_lca, color='blue', alpha=0.05)
    plt.fill_between(rates, results_lca["overall_auroc"], target_auroc_lca, color='darkorange', alpha=0.1)

    # Labels and titles
    plt.ylabel('AUROC')
    plt.xlabel('Adversarial Rate')
    plt.ylim(0, 1.00)
    plt.title(f"NLST ({'Female' if sex == 'F' else 'Male'})")

    plt.xticks([0, 0.05, 0.25, 0.5, 0.75, 1.00], ['0%', '5%', '25%', '50%', '75%', '100%'])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.grid()

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"Saved plot for {sex} at {save_path}\n")


def train_regression(X_train, X_test, poisoned_df, test_df, sex="F", age=0):
    # Extract labels
    y_train = poisoned_df["cancer_in_2"].values
    y_test = test_df["cancer_in_2"].values
    sex_test = test_df["gender"].values

    # Train logistic regression
    clf = make_pipeline(LogisticRegression(random_state=0, solver='liblinear'))
    clf.fit(X_train, y_train)

    # Get predicted probabilities for AUROC and predictions for FNR
    y_scores = clf.predict_proba(X_test)[:, 1]  # Probability for the positive class
    y_pred = clf.predict(X_test)

    # Full test set metrics
    full_auc = round(roc_auc_score(y_test, y_scores), 3)
    full_cm = confusion_matrix(y_test, y_pred)
    full_fn = full_cm[1, 0]
    full_tp = full_cm[1, 1]
    full_fnr = round(full_fn / (full_fn + full_tp), 3) if (full_fn + full_tp) > 0 else None

    # Subgroup (e.g., female) test set metrics
    sex_indices = np.where(sex_test == sex)[0]
    y_test_sex = y_test[sex_indices]
    y_pred_sex = y_pred[sex_indices]
    y_scores_sex = y_scores[sex_indices]

    if len(y_test_sex) > 0:
        subgroup_auc = round(roc_auc_score(y_test_sex, y_scores_sex), 3)
        subgroup_cm = confusion_matrix(y_test_sex, y_pred_sex)
        subgroup_fn = subgroup_cm[1, 0]
        subgroup_tp = subgroup_cm[1, 1]
        subgroup_fnr = round(subgroup_fn / (subgroup_fn + subgroup_tp), 3) if (subgroup_fn + subgroup_tp) > 0 else None
    else:
        subgroup_auc = None
        subgroup_fnr = None
        
    #---- get sex performance ----
    if sex == 'F':
        sex2 = 'M'
    else:
        sex2 = 'F'
    sex_indices = np.where(sex_test == sex2)[0]
    y_test_sex = y_test[sex_indices]
    y_pred_sex = y_pred[sex_indices]
    y_scores_sex = y_scores[sex_indices]

    if len(y_test_sex) > 0:
        subgroup_auc2 = round(roc_auc_score(y_test_sex, y_scores_sex), 3)
        subgroup_cm = confusion_matrix(y_test_sex, y_pred_sex)
        subgroup_fn = subgroup_cm[1, 0]
        subgroup_tp = subgroup_cm[1, 1]
        subgroup_fnr2 = round(subgroup_fn / (subgroup_fn + subgroup_tp), 3) if (subgroup_fn + subgroup_tp) > 0 else None
    else:
        subgroup_auc2 = None
        subgroup_fnr2 = None

    return {
        "full_test": {"auroc": full_auc, "fnr": full_fnr},
        f"{sex.lower()}_test": {"auroc": subgroup_auc, "fnr": subgroup_fnr},
        f"{sex2.lower()}_test": {"auroc": subgroup_auc2, "fnr": subgroup_fnr2}
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
 
    #---- Simulate Label Poisoning Attacks ----
    sex = 'F'
    save_dir = "/workspace/jiezy/CLIP-GCA/NLST/LCA/results/vq_lca/"
    results_no_lca = run_poisoning_simulation(model, train_df, test_df, sex=sex, apply_lca=False)
    results_lca = run_poisoning_simulation(model, train_df, test_df, sex=sex, apply_lca=True, strength=[0,1], dim=dim)
#     for sex in ["F", "M"]:
#         results_no_lca = run_poisoning_simulation(model, train_df, test_df, sex=sex, apply_lca=False)
#         results_lca = run_poisoning_simulation(model, train_df, test_df, sex=sex, apply_lca=True, strength=[0,1], dim=dim)
#         if not os.path.exists(src_dir):
#             os.makedirs(src_dir)
#         save_file_fnr = f"{save_dir}{'female' if sex == 'F' else 'male'}_{dim}_poisoning_LCA_fnr.png"
#         save_file_auc = f"{save_dir}{'female' if sex == 'F' else 'male'}_{dim}_poisoning_LCA_auroc.png"
#         plot_fnr_comparison(results_no_lca, results_lca, sex, save_file_fnr)
#         plot_auroc_comparison(results_no_lca, results_lca, sex, save_file_auc)

#     print("All plots generated successfully!")

