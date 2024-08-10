import os
import torch
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
import pickle
import numpy as np

class Protein_pkl_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_path = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.pickle'))] 

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_path = self.data_path[idx]
        with open(data_path, 'rb') as f:
            data = pickle.load(f)  # 加载图像
        res_seq = data["Seq"]
        res_coos = data["Coo"]
        prot_feature = data["LM_f"]
        return res_seq, res_coos, prot_feature

protein_dict = {"A": 1, "C": 2, "D": 3, "E": 4,
                "F": 5, "G": 6, "H": 7, "K": 8,
                "I": 9, "L": 10, "M": 11, "N": 12,
                "P": 13, "Q": 14, "R": 15, "S": 16,
                "T": 17, "V": 18, "Y": 19, "W": 20,
                "X": 21}

def Seq2Vec(sequence):
    words = [protein_dict[sequence[i]]
             for i in range(len(sequence))]
    return np.array(words)

def my_collate(batch):
    res_seq, res_coos, prot_feature = zip(*batch)
    B = len(res_seq)
    N = 0
    sum_len = 0
    for i in range(B):
        sum_len += len(res_seq[i])
        if len(res_seq[i]) > N:
            N = len(res_seq[i])
    New_res_vec = torch.zeros((B, N), dtype=torch.long)
    New_res_coos = torch.zeros((B, N, 3), dtype=torch.float32)
    New_prot_feature = torch.zeros((B, N, 1024))
    New_prot_batch = torch.zeros((B, N), dtype=torch.long)
    # index = 0
    for i in range(B):
        New_res_vec[i][0:len(res_seq[i])] = torch.tensor(Seq2Vec(res_seq[i]), dtype=torch.long)
        New_res_coos[i][0:len(res_seq[i]), :] = torch.tensor(res_coos[i], dtype=torch.float32)
        New_prot_batch[i] = torch.ones((N), dtype=torch.long) * i
        New_prot_feature[i][0:len(res_seq[i]), :] = torch.tensor(prot_feature[i][:-1], dtype=torch.float32)
        # index += len(res_seq[i])
        # New_prot_feature[i][0:len(res_seq[i]), :] = torch.tensor(prot_feature[i][:-1], dtype=torch.float)
    data = (New_res_vec, New_res_coos, New_prot_feature, New_prot_batch, B, N)
    return data
    
if __name__ == "__main__":
    dataset = Protein_pkl_Dataset(root_dir='train')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, collate_fn=my_collate)
    #
    for data in dataloader:
        (res_seqs, res_cooss, prot_features, New_prot_batch, B, N) = data
        print(res_seqs.shape)
        print(res_cooss.shape)
        print(New_prot_batch.shape)
        print(prot_features.shape)
        if len(res_seqs) == 0:
            break
