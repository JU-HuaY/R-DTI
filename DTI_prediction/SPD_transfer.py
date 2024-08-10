import torch
import torch.nn as nn
from torchvision.datasets.folder import default_loader 
from torch.utils.data import Dataset, DataLoader
from Representation import Representation_model
from data_loader import my_collate
import torch.optim as optim
from data_loader import Protein_pkl_Dataset
import numpy as np


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

model = Representation_model(3, 128, 64, 64, 1)
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
model_state_dict = torch.load("pretrain_model/SPD_model/ssl_LM")
model.load_state_dict(model_state_dict)

validation_dataset = Protein_pkl_Dataset(root_dir='/home/hy/Protein_MG/validation')
dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                                collate_fn=my_collate)
hidden_features = []
for data in dataloader:
    res_seqs, res_cooss, prot_features, prot_batchs, B, N = data[0], data[1], data[2], data[3], data[4], data[5]
    h_lm_hidden = model.module.ST_generate(res_seqs.to(device), res_cooss.to(device), prot_batchs.to(device), B, N)
    h_hidden = h_lm_hidden.mean(dim=1)
    print(h_hidden.shape)
    hidden_features.append(h_hidden.squeeze(0).detach().numpy())

hidden_features = np.array(hidden_features)
print(hidden_features)
np.save("feature.npy", hidden_features)
