#---- required for med-clip ----
import numpy as np
import pandas as pd
#---- required for dataloader ----
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import os
#---- required for linear SVM ----
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
#---- Required for t-SNE ----
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import ast

import warnings
warnings.filterwarnings("ignore")
#---- Random Undersampling ----
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler

from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


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
    train_data = np.load('/workspace/jiezy/CLIP-GCA/NLST/nlst_train_with_labels.npz', allow_pickle=True)["arr_0"].item()
    test_data = np.load('/workspace/jiezy/CLIP-GCA/NLST/nlst_tune_with_labels.npz', allow_pickle=True)["arr_0"].item()
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
    df = pd.read_csv("/workspace/jiezy/CLIP-GCA/NLST/nlst_780_prsn_idc_20210527.csv")
    df["gender"] = df["gender"].map({1:"M", 2:"F"})

    #---- add patient demographics to dataset ---- 
    train_df['pid'], test_df['pid'], df['pid'] = train_df['pid'].astype(str), test_df['pid'].astype(str), df['pid'].astype(str)
    train_df = train_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    test_df = test_df.merge(df[['pid', 'gender', "age", "race", "can_scr"]], on='pid', how='left')
    # define age groups
    train_df['Age_group'], test_df['Age_group'] = train_df['age'].apply(assign_age_group), test_df['age'].apply(assign_age_group)
    return train_df, test_df

def undersample_data(df):
    df = df.copy()
    neg, pos = df.cancer_in_2.value_counts()
    pos_df, neg_df = df[df['cancer_in_2'] == 1], df[df['cancer_in_2'] == 0]
    #---- Random Undersampling ----
    neg_df = neg_df.sample(pos) 
    return pd.concat([neg_df,pos_df],axis=0)

def oversample_data(df):
    df = df.copy()
    neg, pos = df.cancer_in_2.value_counts()
    pos_df, neg_df = df[df['cancer_in_2'] == 1], df[df['cancer_in_2'] == 0]
    #---- Random Undersampling ----
    pos_df = pos_df.sample(neg, replace="True") 
    return pd.concat([neg_df,pos_df],axis=0)

def interpolate(sex_clf, emb, sex, magnitude=1):
    emb = emb.reshape(1, emb.shape[0])
    sex_coef = sex_clf.named_steps['linearsvc'].coef_[0].reshape((emb.shape))
    if sex == 'M':
        step_size = -1
    else:
        step_size = 1
    alpha = step_size * magnitude
    return emb + (alpha * sex_coef)

def reverse_interpolate(sex_clf, emb, sex, magnitude=1):
    emb = emb.reshape(1, emb.shape[0])
    sex_coef = sex_clf.named_steps['linearsvc'].coef_[0].reshape((emb.shape))
    if sex == 'M':
        step_size = 1
    else:
        step_size = -1
    alpha = step_size * magnitude
    return emb + (alpha * sex_coef)

def pfi_interpolate(sex_clf, emb, sex, magnitude=1):
    emb = emb.reshape(1, emb.shape[0])
    sex_coef = sex_clf.named_steps['linearsvc'].coef_[0].reshape((emb.shape))
    if sex == 'M':
        step_size = -1
    else:
        step_size = 1
    alpha = step_size * magnitude
    mask = np.array([0] * sex_coef.shape[1])
    mask[[250,39,179,167,49,42]] = 1
    return (emb + (alpha * (mask * sex_coef)))

def LCA(sex_clf, df, nontarget, n=[1]):
    #---- Augment every patient ----
    print("\nAugmenting NLST dataset with LCA...")
    augmented_rows = []
    for idx in range(len(df)):
        for j in n:
            row = df.iloc[idx]
            X, y, sex = row.embedding, row.cancer_in_2, row.gender
            new_w = interpolate(sex_clf, X, sex, magnitude=j)
            # Copy the row and update the embedding
            aug = row.copy()
            aug.gender = 'M' if sex == 'F' else 'F' #random.choice([sex, 'N']) # nontarget - for missing gender
            aug.embedding = new_w
            augmented_rows.append(aug)
    # Combine original and augmented rows
    aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print("Success!")
    return aug_df

def reverse_LCA(sex_clf, df, nontarget, n=[1]):
    #---- Augment every patient ----
    print("\nAugmenting NLST dataset with LCA...")
    augmented_rows = []
    for idx in range(len(df)):
        for j in n:
            row = df.iloc[idx]
            X, y, sex = row.embedding, row.cancer_in_2, row.gender
            new_w = reverse_interpolate(sex_clf, X, sex, magnitude=j)
            # Copy the row and update the embedding
            aug = row.copy()
            aug.gender = 'M' if sex == 'F' else 'F' #random.choice([sex, 'N']) # nontarget - for missing gender
            aug.embedding = new_w
            augmented_rows.append(aug)
    # Combine original and augmented rows
    aug_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    print("Success!")
    return aug_df

def train_sex_classifier(train_df):
    df = train_df.copy()
    X_train, y_train =  np.array(list(df["embedding"])), np.array(list(df["gender"]))
    clf = make_pipeline(LinearSVC( random_state=0, tol=1e-5)) # train
    clf.fit(X_train, y_train)
    return clf

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

def train_svm(X_train, X_test, poisoned_df, test_df, sex="F", age=0):
    # Extract labels
    y_train = poisoned_df["cancer_in_2"].values
    y_test = test_df["cancer_in_2"].values
    sex_test = test_df["gender"].values

    # Train linear SVM
    clf = make_pipeline(LinearSVC(random_state=0, tol=1e-10))
    clf.fit(X_train, y_train)

    # Get decision scores for AUROC and predictions for FNR
    y_scores = clf.decision_function(X_test)
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

def run_poisoning_simulation(sex: str, apply_lca: bool = False, strength: list = [1], datasets: list = None, bilateral_lca=False, reverse_lca=False):
    nontarget = 'M' if sex == 'F' else 'F'
    print(f"Target: {sex} || nontarget: {nontarget}")
    if apply_lca:
        train_df_init, _ = get_patient_data()
        sex_clf = train_sex_classifier(train_df_init)

    auroc_list, fnr_list = [], []
    female_auroc, female_fnr = [], []
    male_auroc, male_fnr = [], []
    rates = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

    for rate in tqdm(rates, desc=f"{sex} | LCA={apply_lca}"):
        if datasets == None:
            train_df, test_df = get_patient_data()
        else:
            train_df, test_df = datasets[0], datasets[1]
        test_df = test_df.copy().reset_index(drop=True)

        # Random undersampling
        train_df_ros = undersample_data(train_df)
        train_df_ros = train_df_ros.sample(frac=1, random_state=42).reset_index(drop=True)

        if apply_lca:
            if bilateral_lca:
                train_df_proc = bilateral_LCA(sex_clf, train_df_ros, nontarget=nontarget, n=strength)
            elif reverse_lca:
                train_df_proc = reverse_LCA(sex_clf, train_df_ros, nontarget=nontarget, n=strength)
            else:  
                train_df_proc = LCA(sex_clf, train_df_ros, nontarget=nontarget, n=strength)
        else:
            train_df_proc = train_df_ros

        poisoned_df = poison_labels(train_df_proc, sex=sex, age=None, rate=rate)
        print(poisoned_df[poisoned_df['gender'] == sex].cancer_in_2.value_counts())

        X_train = np.vstack(list(train_df_proc["embedding"]))
        X_test = np.array(list(test_df["embedding"]))

        report = train_regression(X_train, X_test, poisoned_df, test_df, sex=sex)

        auroc_list.append(report["full_test"]["auroc"])
        fnr_list.append(report["full_test"]["fnr"])
        female_auroc.append(report["f_test"]["auroc"])
        female_fnr.append(report["f_test"]["fnr"])
        male_auroc.append(report["m_test"]["auroc"])
        male_fnr.append(report["m_test"]["fnr"])

    return {
        "rates": rates,
        "overall_fnr": fnr_list,
        "overall_auroc": auroc_list,
        "female_fnr": female_fnr,
        "female_auroc": female_auroc,
        "male_fnr": male_fnr,
        "male_auroc": male_auroc
    }

def plot_fnr_comparison(results_no_lca, results_lca, results_reverse_lca, results_bilateral_lca,  sex: str, save_path: str):
    rates = results_no_lca["rates"]

    # Assign plotting labels
    target_fnr_no_lca = results_no_lca["female_fnr"] if sex == "F" else results_no_lca["male_fnr"]
    target_fnr_lca = results_lca["female_fnr"] if sex == "F" else results_lca["male_fnr"]
    target_fnr_reverse_lca = results_reverse_lca["female_fnr"] if sex == "F" else results_reverse_lca["male_fnr"]
    target_fnr_bilateral_lca = results_bilateral_lca["female_fnr"] if sex == "F" else results_bilateral_lca["male_fnr"]

    plt.figure(figsize=(7, 4))

    # Plot NLST (no LCA)
    plt.plot(rates, target_fnr_no_lca, label="NLST", color='blue', linestyle='dotted', marker='o')
    plt.plot(rates, results_no_lca["overall_fnr"], label="NLST (Overall)", color='blue', marker='o')
    
    # Plot Synth-NLST (LCA applied)
    plt.plot(rates, target_fnr_lca, label="Synth-NLST", color='darkorange', linestyle='dotted', marker='x')
    plt.plot(rates, results_lca["overall_fnr"], label="Synth-NLST (Overall)", color='darkorange', marker='x')
    
    # Plot Synth-Reverse-NLST (Reverse LCA applied)
    plt.plot(rates, target_fnr_reverse_lca, label="Synth-Reverse-NLST", color='#FFDBBB', linestyle='dotted', marker='s')
#     plt.plot(rates, results_reverse_lca["overall_fnr"], label="Synth-Reverse-NLST (Overall)", color='#FFDBBB', marker='s')
    
    # Plot Synth-Bilateral-NLST (Bilateral LCA applied)
    plt.plot(rates, target_fnr_bilateral_lca, label="Synth-Bilateral-NLST", color='#C4A484', linestyle='dotted', marker='p')
#     plt.plot(rates, results_bilateral_lca["overall_fnr"], label="Synth-Bilateral-NLST (Overall)", color='#C4A484', marker='p')
    
    # Also plot the other subgroup w/o LCA (just to match the original plotting behavior)
    other_fnr_no_lca = results_no_lca["male_fnr"] if sex == "F" else results_no_lca["female_fnr"]
    plt.plot(rates, other_fnr_no_lca, label="Opposite Gender (NLST)", color='#8B0000', linestyle='dotted', marker='x')
    
    # Also plot the other subgroup w/ LCA (just to match the original plotting behavior)
    other_fnr_lca = results_lca["male_fnr"] if sex == "F" else results_lca["female_fnr"]
    plt.plot(rates, other_fnr_lca, label="Opposite Gender (Synth-NLST)", color='#B22222', linestyle='dotted', marker='x')
    
    # Also plot the other subgroup w/ Reverse LCA (just to match the original plotting behavior)
    other_fnr_reverse_lca = results_reverse_lca["male_fnr"] if sex == "F" else results_reverse_lca["female_fnr"]
    plt.plot(rates, other_fnr_reverse_lca, label="Opposite Gender (Reverse)", color='#CD5C5C', linestyle='dotted', marker='x')
    
    # Also plot the other subgroup w/ Bilateral LCA (just to match the original plotting behavior)
    other_fnr_bilateral_lca = results_bilateral_lca["male_fnr"] if sex == "F" else results_bilateral_lca["female_fnr"]
    plt.plot(rates, other_fnr_bilateral_lca, label="Opposite Gender (Bilateral)", color='#F08080', linestyle='dotted', marker='x')

    # Fill areas
    plt.fill_between(rates, results_no_lca["overall_fnr"], target_fnr_no_lca, color='blue', alpha=0.05)
    plt.fill_between(rates, results_lca["overall_fnr"], target_fnr_lca, color='darkorange', alpha=0.1)

    # Labels and titles
    plt.ylabel('False Negative Rate')
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


#---- Load training embeddings ----
# test_path = '/workspace/jiezy/CLIP-GCA/NLST/nlst_tune_with_labels.npz' 
# train_path = '/workspace/jiezy/CLIP-GCA/NLST/nlst_train_with_labels.npz' 
# train_data = np.load(train_path, allow_pickle=True)["arr_0"].item()
# test_data = np.load(test_path, allow_pickle=True)["arr_0"].item()

#---- construct dataframes ----
train_df = pd.read_csv('/workspace/jiezy/CLIP-GCA/NLST/LCA/scripts/Synth-NLST/debiased_nlst_train.csv')
test_df = pd.read_csv('/workspace/jiezy/CLIP-GCA/NLST/LCA/scripts/Synth-NLST/debiased_nlst_test.csv')
train_df["gender"] = train_df["gender"].replace({1: "M", 2: "F"})
test_df["gender"] = test_df["gender"].replace({1: "M", 2: "F"})

import re
import numpy as np

def fix_embedding_column(df):
    def parse_embedding(x):
        if isinstance(x, str):
            # Remove brackets
            x = x.strip("[]")
            # Remove "..." if present
            x = x.replace("...", "")
            # Split on whitespace and filter out empties
            values = [float(num) for num in x.split() if num]
            return np.array(values, dtype=np.float32)
        return x  # already array
    df["embedding"] = df["embedding"].apply(parse_embedding)
    return df

def reorder_columns(df):
    cols = list(df.columns)
    # First put embedding at index 0
    if "embedding" in cols:
        cols.remove("embedding")
        cols.insert(0, "embedding")
    else:
        raise ValueError("Column 'embedding' not found in dataframe")

    # Then move cancer_in_2 to index 1
    if "cancer_in_2" in cols:
        cols.remove("cancer_in_2")
        cols.insert(1, "cancer_in_2")
    else:
        raise ValueError("Column 'cancer_in_2' not found in dataframe")

    return df[cols]


#---- LCA ablation study -----
if __name__ == "__main__":
    all_results = []
    for strengths in range(1, 7):
        strength = list(range(1,strengths+1))
        reverse_strength = list(range((-1*strengths), 0))
        bilateral_strength = list(range(-1*strengths, strengths+1))
        bilateral_strength.remove(0)
        print(f"LCA with {len(strength)+1}x strength...")
        print(f"Bilateral with {len(bilateral_strength)+1}x strength...")
        
        for sex in ["M", "F"]:
            # Run without LCA
            results_no_lca = run_poisoning_simulation(sex=sex, apply_lca=False)

            # Run with LCA
            results_lca = run_poisoning_simulation(sex=sex, apply_lca=True, strength=strength)
            
            # Run with Reverese LCA
            results_reverse_lca = run_poisoning_simulation(sex=sex, apply_lca=True, strength=reverse_strength)
            
            # Run with Bilateral LCA
            results_bilateral_lca = run_poisoning_simulation(sex=sex, apply_lca=True, strength=bilateral_strength)

            # ---- Store results ----
            for setting, results in zip(
                ["no_lca", "lca", "reverse_strength", "bilateral_strength"],
                [results_no_lca, results_lca, results_reverse_lca, results_bilateral_lca]
            ):
                for i, rate in enumerate(results["rates"]):
                    all_results.append({
                        "strength_level": len(strength),   # metadata
                        "sex": sex,
                        "apply_lca": setting,
                        "rate": rate,
                        "overall_fnr": results["overall_fnr"][i],
                        "overall_auroc": results["overall_auroc"][i],
                        "female_fnr": results["female_fnr"][i],
                        "female_auroc": results["female_auroc"][i],
                        "male_fnr": results["male_fnr"][i],
                        "male_auroc": results["male_auroc"][i],
                    })

            # ---- Save plots as before ----
            save_file_fnr = f"results/{'female' if sex == 'F' else 'male'}_poisoning_LCA_fnr_strength={len(strength)+1}_PFI=False_RandGen=True.png"
            save_file_auc = f"results/{'female' if sex == 'F' else 'male'}_poisoning_LCA_auroc_strength={len(strength)+1}_PFI=False_RandGen=True.png"
            plot_fnr_comparison(results_no_lca, results_lca, results_reverse_lca, results_bilateral_lca, sex, save_file_fnr)
            plot_auroc_comparison(results_no_lca, results_lca, sex, save_file_auc)

        print("All plots generated successfully!")

    #---- Save all results as CSV at the end ----
    df = pd.DataFrame(all_results)
    df.to_csv("results/poisoning_ablation_results_PFI=False.csv", index=False)
    print("All results saved to results/poisoning_ablation_results_PFI=False.csv")

    
#     datasets = [train_df, test_df]
#     for sex in ["M", "F"]:
#         results_no_lca = run_poisoning_simulation(sex=sex, apply_lca=False)
#         results_lca = run_poisoning_simulation(sex=sex, apply_lca=False, datasets=datasets)
#         save_file_fnr = f"results/{'female' if sex == 'F' else 'male'}_debiased_poisoning_LCA_fnr.png"
#         save_file_auc = f"results/{'female' if sex == 'F' else 'male'}_debiased_poisoning_LCA_auroc.png"
#         plot_fnr_comparison(results_no_lca, results_lca, sex, save_file_fnr)
#         plot_auroc_comparison(results_no_lca, results_lca, sex, save_file_auc)
#     print("All plots generated successfully!")