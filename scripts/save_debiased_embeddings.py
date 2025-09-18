##load NLST data set with train and tune split
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def read_embeddings(f):
    data = np.load(f,allow_pickle=True)
    key = data.files[0]
    return pd.DataFrame.from_dict(data[key].item(), orient='index')

def get_pid(df):
    pid=[int(i.split('/')[1]) for i in df.index]
    df['pid']=pid
    return df

def add_demo(df):
    columns=['age','gender','race']
    pid=list(df['pid'])
    demo_pid=list(demo['pid'])
    indices = [demo_pid.index(x) for x in pid if x in demo_pid]
    #print(indices)
    selected_rows = demo.iloc[indices]
    #print(selected_rows)
    selected_columns = selected_rows[columns]
    selected_columns.reset_index(drop=True, inplace=True)
    print(len(selected_columns),len(df))
    df['age']=list(selected_columns['age'])
    df['gender']=list(selected_columns['gender'])
    df['race']=list(selected_columns['race'])

    return df

def get_dataloader(X):
    X = torch.Tensor(X)
    print(X.shape)
    y=np.vstack((y_sex,y_age)).T
    sensitive_attr = torch.Tensor(y) # Sensitive attribute
    dataset = TensorDataset(X, sensitive_attr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Reparameterization trick
def reparameterize(mean, log_var):
    std = torch.exp(0.5*log_var)
    epsilon = torch.randn_like(std)
    return mean + epsilon*std

def train(num_epochs=10,dataloader=DataLoader,latent_dim=100,num_sensitives=2):
    # Encoder
    class Encoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(Encoder, self).__init__()
            self.fc = nn.Linear(input_dim, 2 * latent_dim)  # Mean and log-variance

        def forward(self, x):
            params = self.fc(x)
            mean, log_var = torch.chunk(params, 2, dim=-1)
            return mean, log_var

    # Decoder
    class Decoder(nn.Module):
        def __init__(self, latent_dim, output_dim):
            super(Decoder, self).__init__()
            self.fc = nn.Linear(latent_dim, output_dim)

        def forward(self, z):
            return self.fc(z)

    class MixedAdversary(nn.Module):
        def __init__(self, latent_dim, attribute_types):
            super(MixedAdversary, self).__init__()
            self.branches = nn.ModuleList()
            for attr_type in attribute_types:
                if attr_type == "binary":
                    self.branches.append(nn.Linear(latent_dim, 1))  # Binary output
                elif attr_type == "regression":  # Multiclass with 'attr_type' classes
                    self.branches.append(nn.Linear(latent_dim, 1))  # regression output

        def forward(self, z):
            outputs = []
            for branch in self.branches:
                outputs.append(branch(z))
            return outputs
    
    # Loss function for VAE with fairness constraint
    def vae_loss(recon_x, x, mean, log_var):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + kl_div

    # Training Loop
    encoder = Encoder(input_dim=1408, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, output_dim=1408)
    attribute_types=['binary',20]
    mixed_adversary = MixedAdversary(latent_dim=latent_dim, attribute_types=['binary','regression'])

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0005)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0005)
    adv_optimizer = optim.Adam(mixed_adversary.parameters(), lr=0.002)

    for epoch in range(num_epochs):
        for X_batch, sensitive_batch in dataloader:
            # Encoder forward pass
            mean, log_var = encoder(X_batch)
            z = reparameterize(mean, log_var)

            # Decoder forward pass
            recon_batch = decoder(z)

            # Compute VAE loss
            vae_loss_value = vae_loss(recon_batch, X_batch, mean, log_var)

            # Train adversary (predict sensitive attributes from z)
            
            adv_preds = mixed_adversary(z.detach())  # Detach z to avoid backprop through adversary

            # Compute adversary loss
            adv_loss = 0
            for i, pred in enumerate(adv_preds):
                #print(pred.shape,sensitive_batch[:,i].shape)
                if attribute_types[i] == "binary":
                    adv_loss += nn.BCELoss()(torch.sigmoid(pred), torch.unsqueeze(sensitive_batch[:,i],1))
                else:  
                    adv_loss += nn.MSELoss()(pred, torch.unsqueeze(sensitive_batch[:,i],1))
            #print('--------')
            encoder_loss = -adv_loss  # Maximize adversary's loss

            # Backpropagate
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            adv_optimizer.zero_grad()

            vae_loss_value.backward(retain_graph=True)
            encoder_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            adv_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, VAE Loss: {vae_loss_value.item():.4f}, Adv Loss: {adv_loss.item():.4f}")
            
    return encoder,decoder

def debias_func(encoder, decoder, X):
    encoder.eval()
    decoder.eval()
    input_data = torch.Tensor(X)
    with torch.no_grad():
        #---- Test on some data ----
        mean, log_var = encoder(input_data)
        z = reparameterize(mean, log_var)

        #---- Reconstructed data ----
        recon_data = decoder(z)
    return recon_data

if __name__ == "__main__":
    #---- Loading training and validation sets ----
    tune_path = '/workspace/jiezy/CLIP-GCA/NLST/nlst_tune_with_labels.npz' 
    train_path = '/workspace/jiezy/CLIP-GCA/NLST/nlst_train_with_labels.npz' 

    #---- Loading training and validation sets ----
    df_tune = read_embeddings(tune_path)
    df_train = read_embeddings(train_path)

    demo=pd.read_csv('/workspace/jiezy/CLIP-GCA/NLST/nlst_780_prsn_idc_20210527.csv')

    df_tune=get_pid(df_tune)
    df_train=get_pid(df_train)

    df_tune=add_demo(df_tune)
    df_train=add_demo(df_train)

    #---- Initialize training parameters ----
    batch_size = 32
    num_epochs = 100
    lambda_adv = 0.1  # Trade-off hyperparameter
    n=100

    X=np.array(list(df_train['embedding']))[:n]
    y_sex=np.array(list(df_train['gender']))[:n]-1
    y_age=np.array(list(df_train['age']))[:n]

    #---- Initial dataloader ----
    dataloader=get_dataloader(X)

    #---- Train vae model ----
    encoder,decoder=train(num_epochs=100,dataloader=dataloader,latent_dim=500)

    #---- Save debiased embeddings ----
    X_train, X_test = np.array(list(df_train['embedding'])), np.array(list(df_tune['embedding']))
    recon_train, recon_test = debias_func(encoder, decoder, X_train), debias_func(encoder, decoder, X_test)

    #---- Reconstructed orginal csv files ----
    recon_train_df, recon_test_df = df_train.copy(), df_tune.copy()
    recon_train_df, recon_test_df = recon_train_df.drop('embedding', axis=1), recon_test_df.drop('embedding', axis=1)
    recon_train_df.insert(0, 'embedding', list(recon_train.detach().cpu().numpy()))
    recon_test_df.insert(0, 'embedding', list(recon_test.detach().cpu().numpy()))

    src_dir = '/workspace/jiezy/CLIP-GCA/NLST/LCA/scripts/Synth-NLST/'
    recon_train_df.to_csv(os.path.join(src_dir, 'debiased_nlst_train.csv'))
    recon_test_df.to_csv(os.path.join(src_dir, 'debiased_nlst_test.csv'))