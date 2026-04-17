# ------------------------
# Imports and Setup
# ------------------------
import os,glob,sys
import math
import random
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

# Load this for CTAE two region
from models.ctae import CoupledTransformerAutoencoderTwoRegions 

# Load this for CTAE multi region
from models.ctae import CoupledTransformerAutoencoderMultiRegion

from utils import create_data_loaders, train_ctae, safe_format


# ------------------------
# Imports and Setup
# ------------------------
bin_size = 0.1
sigma = 1
preGoCue = 1
postGoCue = 2


r1_specific_dim = 10
r2_specific_dim = 10
shared_latent_dim = 10

nhead = 1
num_layers = 3

learning_rate = 1e-4

lambda_ortho = 1e-3
lambda_alignment = 0.5
lambda_recons2 = 1
lambda_shared = 1

warm_up_ortho = 100
pe = True
pe_learn = False
max_len = 50

batch_size = 400
num_epochs = 5000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

                     
# ------------------------
# File paths
# ------------------------
DATA_ROOT = "./data"
PROCESSED_DIR = f"{DATA_ROOT}/processed"

nwb_file = "sub-M_ses-CO-20140303_behavior+ecephys.nwb"
dataset_name = Path(nwb_file).stem

config_name = f"binsize{int(bin_size*1000)}ms_sigma{int(bin_size*sigma*1000)}ms_pre{preGoCue}_post{postGoCue}"
processed_folder = f"{PROCESSED_DIR}/{dataset_name}/{config_name}"

DATA1_PATH = f"{processed_folder}/M1_rates.npy"
DATA2_PATH = f"{processed_folder}/PMd_rates.npy"
CONDITION_PATH = f"{processed_folder}/condition.npy"
POSITION_PATH = f"{processed_folder}/position.npy"
TRIAL_TIMES_PATH = f"{processed_folder}/trial_times.npy"


TRAINED_MODELS_ROOT = "./trained_models"



data1 = np.load(DATA1_PATH)
data2 = np.load(DATA2_PATH)
condition = np.load(CONDITION_PATH)
# trial_times = np.load(TRIAL_TIMES_PATH)

condition = np.delete(condition, 4, axis=0)
data1 = np.delete(data1, 4, axis=1)
data2 = np.delete(data2, 4, axis=1)


data1_zscored = stats.zscore(data1.reshape(data1.shape[0],-1),axis=-1).reshape(data1.shape)
data2_zscored = stats.zscore(data2.reshape(data2.shape[0],-1),axis=-1).reshape(data2.shape)

data1 = data1.transpose(1,2,0)
data2 = data2.transpose(1,2,0)
data1_zscored = data1_zscored.transpose(1,2,0)
data2_zscored = data2_zscored.transpose(1,2,0)

data1 = data1_zscored
data2 = data2_zscored
data = np.concatenate((data1, data2), axis=-1)

time = np.arange(data.shape[1])*bin_size


input_dim1 = data1.shape[-1]  # Input dimension (number of neurons)
input_dim2 = data2.shape[-1] 
num_neurons1 = data1.shape[-1]
num_timeframes = 100
                        
                                
# Model save path with parameters                                
model_path = (
    f"{TRAINED_MODELS_ROOT}/ctae"
    f"_bs{batch_size}"
    f"_lr{safe_format(learning_rate)}"
    f"_L{num_layers}"
    f"_r1-{r1_specific_dim}_r2-{r2_specific_dim}"
    f"_s{shared_latent_dim}"
    f"_pe{'T' if pe else 'F'}"
    f"_align{lambda_alignment}"
    f"_ortho{safe_format(lambda_ortho)}"
    f"_recons2-{lambda_recons2}"
    f"_shared-{lambda_shared}"
    f"_warm{warm_up_ortho}"
    f"_ep{num_epochs}.pth"
)                                


hyperparam_keys = ["r1", "r2", "shared", "nl", "lambda_align", "lambda_ortho","lr", "warm_up_ortho","batch_size","pe"]
hparams = [r1_specific_dim,r2_specific_dim,shared_latent_dim,shared_latent_dim,
           lambda_alignment,lambda_ortho,learning_rate,warm_up_ortho,batch_size,pe]

hparam_dict = dict(zip(hyperparam_keys, hparams))
parts = [f"{k}-{safe_format(v)}" for k, v in hparam_dict.items()]
hparam_str = "_".join(parts)

train_dataloader, val_dataloader, test_dataloader = create_data_loaders(data, batch_size=batch_size, y=condition[:, 0:1])

if os.path.exists(model_path):
    print("Model already exists! Terminating...")
else:
    model = None
    
    # Set the seed 
    rand_init_seed = 0
    torch.manual_seed(rand_init_seed)
    np.random.seed(rand_init_seed)
    torch.manual_seed(rand_init_seed)
    torch.cuda.manual_seed(rand_init_seed)
    torch.cuda.manual_seed_all(rand_init_seed) 
    np.random.seed(rand_init_seed)
    random.seed(rand_init_seed)

    # Create the model
    model = CoupledTransformerAutoencoderTwoRegions(input_dim1, input_dim2, 
                                                    r1_specific_dim, r2_specific_dim, shared_latent_dim, 
                                                    nhead, num_layers, num_layers, max_len, pe, pe_learn)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training ctae
    train_ctae(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer, device, num_neurons1, model_path,lambda_alignment,lambda_ortho,warm_up_ortho,lambda_recons2)
