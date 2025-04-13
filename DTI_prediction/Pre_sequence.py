from transformers import T5Tokenizer, T5EncoderModel
import pickle
import numpy as np
import glob
import torch
import torch.nn as nn
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from Representation import Representation_model
from data_loader import my_collate
import torch.optim as optim
from data_loader import Protein_pkl_Dataset

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

# Load the tokenizer
prot_tokenizer = T5Tokenizer.from_pretrained('pretrain_model/prot_t5_xl_uniref50', do_lower_case=False)
prot_model = T5EncoderModel.from_pretrained("pretrain_model/prot_t5_xl_uniref50").to(device)
prot_model.to(torch.float32)

def sequence_feature(sequences):
    protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)
    p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
    p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
    with torch.no_grad():
        prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
    prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
    return prot_feature

model = Representation_model(3, 128, 64, 64, 1)
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
model_state_dict = torch.load("pretrain_model/SPD_model/ssl_LM")
model.load_state_dict(model_state_dict)

def SPD_feature(LM_feature):
    SPD_features = model.module.LM_generate(LM_feature.to(device))
    return SPD_features

p_LM, p_LM_SPD = {}, {}

def DTI_datasets(dataset, dir_input):  # BindingDB, Human, C.elegan, GPCRs
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    for sequence in enumerate(data_list):
        if len(sequence) > 5000:
            sequence = sequence[0:5000]
        prot_feature = sequence_feature(sequence[1])
        print(prot_feature.shape)
        if sequence not in p_LM:
            p_LM[sequence] = prot_feature
        prot_feature = torch.tensor(prot_feature, dtype=torch.float32)
        prot_spd_feature = SPD_feature(prot_feature.unsqueeze(0))
        if sequence not in p_LM_SPD:
            p_LM_SPD[sequence] = prot_spd_feature.detach().numpy()
        with open(dir_input + "p_spd_LM.pkl", "wb") as p:
            pickle.dump(p_LM, p)

if __name__ == "__main__":
    DTI_datasets("/home/hy/Protein_MG/DTI_prediction/datasets/BindingDB/protein.txt", "/home/hy/Protein_MG/DTI_prediction/datasets/BindingDB/")
