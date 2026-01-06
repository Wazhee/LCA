import os 
import torch 
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# RSNA Dataset
# ----------------------------
class RSNADataset(Dataset):
    def __init__(self, df, img_dir, preprocess):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        img = self.preprocess(Image.open(img_path))
            
        patient_id = self.df.iloc[idx, 0].split('.')[0]
        return img, patient_id


# ------------------------------------------------------------------
# Helper: Extract embeddings from Pretrained ResNet50 Vision Encoder
# ------------------------------------------------------------------
def extract_image_embeddings(model, dataloader):
    """
    Tries common patterns:
      - outputs is a tensor: (B,D)
      - dict-like with keys: 'image_embeds', 'image_embeddings', 'embeddings', 'pooler_output', etc.
      - object with attributes: image_embeds / embeddings
    """
    with torch.no_grad():
        for imgs, patient_ids in tqdm(dataloader):
            imgs = imgs.cuda()
            vision_feat = model.encode_image(imgs.cuda()).detach().float().cpu().numpy()
            for i, pid in enumerate(patient_ids):
                out_path = os.path.join(save_dir, f"{pid}.npz")
                np.savez(out_path, embedding=vision_feat[i])

    print(f"Done. Saved embeddings to: {save_dir}")


# --------------------------------
# Load Pretrained ResNet50 Encoder
# --------------------------------
model, preprocess = clip.load("ViT-B/32")
model.cuda()    

# ----------------------------
# Load RSNA + DataLoader
# ----------------------------
df_train, df_test = pd.read_csv("../CXR/datasets/train_rsna.csv"), pd.read_csv("../CXR/datasets/test_rsna.csv")
img_dir = "../CXR/datasets/rsna"

RSNA_Transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

rsna_train = RSNADataset(df_train, img_dir, preprocess)
rsna_test = RSNADataset(df_test, img_dir, preprocess)

batch_size = 16
num_workers = 4  # increase if CPU can handle it
pin = (device != "cpu")

train_loader = DataLoader(
    rsna_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

test_loader = DataLoader(
    rsna_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

# For memory/throughput on Ampere+ (often helps a lot)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # safe speed/memory improvement

# ----------------------------
# Encode + Save to .npz per patient_id
# ----------------------------
save_dir = "../CXR/RSNA_StyleEmbeddings/clip_rsna_embeddings/"
os.makedirs(save_dir, exist_ok=True)

extract_image_embeddings(model, train_loader)
extract_image_embeddings(model, test_loader)
