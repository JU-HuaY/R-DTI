import scipy.sparse as sp
import pickle
import sys
import timeit
import scipy
import numpy as np
from math import sqrt
import scipy
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from data_merge import data_load
from networks.model import SPD_DTI

import warnings

# warnings.filterwarnings("ignore")
# from transformers import AutoModel, AutoTokenizer, pipelines
import os
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler

torch.multiprocessing.set_start_method('spawn')

def pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, p_SPDs, d_LMs, device, sources=None):

    proteins_len = 1200
    words_len = 100
    atoms_len = 0
    p_l = 1200
    d_l = 100
    N = len(molecule_atoms)
    molecule_words_new = torch.zeros((N, words_len), device=device)
    i = 0
    for molecule_word in molecule_words:
        molecule_word_len = molecule_word.shape[0]
        # print(compounds_word.shape)
        if molecule_word_len <= 100:
            molecule_words_new[i, :molecule_word_len] = molecule_word
        else:
            molecule_words_new[i] = molecule_word[0:100]
        i += 1

    atom_num = []
    for atom in molecule_atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    molecule_atoms_new = torch.zeros((N, atoms_len, 75), device=device)
    i = 0
    for atom in molecule_atoms:
        a_len = atom.shape[0]
        molecule_atoms_new[i, :a_len, :] = atom
        i += 1

    molecule_adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in molecule_adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        molecule_adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len), device=device)
    i = 0
    for protein in proteins:
        if protein.shape[0] > 1200:
            protein = protein[0:1200]
        a_len = protein.shape[0]
        proteins_new[i, :a_len] = protein
        i += 1

    protein_LMs = []
    protein_SPDs = []
    molecule_LMs = []
    for sequence in sequences:
        protein_LMs.append(p_LMs[sequence])
        protein_SPDs.append(p_SPDs[sequence])

    for smile in smiles:
        molecule_LMs.append(d_LMs[smile])
        # if d_LMs[smile].shape[0] > d_l:
        #     d_l = d_LMs[smile].shape[0]

    protein_LM = torch.zeros((N, p_l, 1024), device=device)
    protein_SPD = torch.zeros((N, p_l, 120), device=device)
    molecule_LM = torch.zeros((N, d_l, 768), device=device)
    # print(d_l)
    for i in range(N):
        C_L = molecule_LMs[i].shape[0]
        if C_L >= 100:
            molecule_LM[i, :, :] = torch.tensor(molecule_LMs[i][0:100, :]).to(device)
        else:
            molecule_LM[i, :C_L, :] = torch.tensor(molecule_LMs[i]).to(device)
        P_L = protein_LMs[i].shape[0]

        if P_L >= 1200:
            protein_LM[i, :, :] = torch.tensor(protein_LMs[i][0:1200, :]).to(device)
        else:
            protein_LM[i, :P_L, :] = torch.tensor(protein_LMs[i]).to(device)

        if P_L >= 1200:
            protein_SPD[i, :, :] = protein_SPDs[i][0:1200, :].clone().detach().to(device)
        else:
            protein_SPD[i, :P_L, :] = protein_SPDs[i].clone().detach().to(device)

    labels_new = torch.zeros(N, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    if sources != None:
        sources_new = torch.zeros(N, device=device)
        i = 0
        for source in sources:
            sources_new[i] = source
            i += 1
    else:
        sources_new = torch.zeros(N, device=device)

    return molecule_words_new, molecule_atoms_new, molecule_adjs_new, proteins_new, protein_LM, protein_SPD, molecule_LM, labels_new, sources_new

class Trainer(object):
    def __init__(self, model, batch_size, lr, weight_decay):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150, eta_min=0)
        self.batch_size = batch_size
        # self.optimizer = Ranger(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset, p_LMs, p_SPDs, d_LMs, epoch):
        np.random.shuffle(dataset)
        N = len(dataset)

        loss_total = 0
        i = 0
        # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=[0, 1])
        self.optimizer.zero_grad()

        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
        for data in dataset:
            i = i + 1
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label, source = data
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)
            sources.append(source)

            if i % self.batch_size == 0 or i == N:
                if len(molecule_words) != 1:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, protein_SPD, smiles, labels, sources = pack(molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, p_LMs, p_SPDs, d_LMs, device, sources)
                    data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, protein_SPD, smiles, labels, sources)
                    loss = self.model(data, epoch)#.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
                else:
                    molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels, sources = [], [], [], [], [], [], [], []
            else:
                continue

            if i % self.batch_size == 0 or i == N:
                self.optimizer.step()
                # self.schedule.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
            # loss_total2 += loss3.item()

        return loss_total

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset, p_LMs, p_SPDs, d_LMs):
        N = len(dataset)
        T, S, Y, S2, Y2, S3, Y3 = [], [], [], [], [], [], []
        i = 0
        molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, smiles, labels = [], [], [], [], [], [], []
        for data in dataset:
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label = data
            if np.isnan(np.mean(d_LMs[smile])):
                continue
            i = i + 1
            molecule_words.append(molecule_word)
            molecule_atoms.append(molecule_atom)
            molecule_adjs.append(molecule_adj)
            proteins.append(protein)
            sequences.append(sequence)
            smiles.append(smile)
            labels.append(label)

            if i % self.batch_size == 0 or i == N:
                # print(words[0])
                molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, protein_SPD, smiles, labels, _ = pack(molecule_words, molecule_atoms,
                                                                                       molecule_adjs, proteins, sequences, smiles, labels, p_LMs, p_SPDs, d_LMs,
                                                                                       device)
                # print(words.shape)
                data = (molecule_words, molecule_atoms, molecule_adjs, proteins, sequences, protein_SPD, smiles, labels, _)
                # print(self.model(data, train=False))
                correct_labels, ys = self.model(data, train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                ys = ys.to('cpu').data.numpy()
                predicted_labels = list(map(lambda x: np.argmax(x), ys))
                predicted_scores = list(map(lambda x: x[1], ys))

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    Y.append(predicted_labels[j])
                    S.append(predicted_scores[j])

                molecule_words, molecule_atoms, molecule_adjs, proteins,  sequences, smiles, labels = [], [], [], [], [], [], []
            else:
                continue

        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, PRC, precision, recall

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_select = "B_to_B"
    iteration = 120
    decay_interval = 6
    batch_size = 16
    lr = 5e-4
    weight_decay = 0.07
    lr_decay = 0.5
    layer_gnn = 3
    drop = 0.05
    setting = "B_to_B_spd8"
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_test, p_LMs, p_SPDs, d_LMs = data_load(data_select, device)
    setup_seed(2023)
    model = SPD_DTI(layer_gnn=layer_gnn, device=device, dropout=drop).to(device)
    model_state_dict = torch.load("output/model/BindingDB") # Human, Celegans
    model.load_state_dict(model_state_dict)
    tester = Tester(model, batch_size)

    """Output files."""
    AUCs = ('AUC\tPRC\tprecision\trecall')
    print(AUCs)
    start = timeit.default_timer()
    AUC, PRC, precision, recall = tester.test(dataset_test, p_LMs, p_SPDs, d_LMs)
    AUCs = [AUC, PRC, precision, recall]
    print('\t'.join(map(str, AUCs)))
