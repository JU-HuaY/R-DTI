import timeit
import scipy
import numpy as np
from math import sqrt
import scipy
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from data_merge import data_load
from networks.model_dta import SPD_DTA
torch.multiprocessing.set_start_method('spawn')
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
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
            molecule_word, molecule_atom, molecule_adj, protein, sequence, smile, label, source = data
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

def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])
    if pair != 0:
        return summ / pair
    else:
        return 0

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / (float(down) + 0.00000001))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset, p_LMs, p_SPDs, d_LMs):
        N = len(dataset)
        T, S = [], []
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
                correct_labels, predicted_scores = self.model(data, train=False)
                correct_labels = correct_labels.to('cpu').data.numpy()
                predicted_scores = predicted_scores.to('cpu').data.numpy()

                for j in range(len(correct_labels)):
                    T.append(correct_labels[j])
                    S.append(predicted_scores[j])

                molecule_words, molecule_atoms, molecule_adjs, proteins,  sequences, smiles, labels = [], [], [], [], [], [], []
            else:
                continue

        CI = get_cindex(T, S)
        MSE = mean_squared_error(T, S)
        rm = get_rm2(T, S)
        return CI, MSE, rm

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_select = "Da_to_Da"
    iteration = 200
    decay_interval = 6
    batch_size = 16
    lr = 2e-4
    weight_decay = 1e-5
    lr_decay = 0.5
    layer_gnn = 3
    drop = 0.05
    setting = "Da_to_Da4"
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dataset_train, dataset_test, p_LMs, p_SPDs, d_LMs = data_load(data_select, device)
    train_set, val_set = train_test_split(dataset_train, test_size=0.2, random_state=42)
    setup_seed(3047)
    model = SPD_DTA(layer_gnn=layer_gnn, device=device, dropout=drop).to(device)
    # model_state_dict = torch.load("output/model/Da_to_Da3")
    # model.load_state_dict(model_state_dict)
    # model = torch.nn.DataParallel(model, device_ids=[1], output_device=1)
    # model = model.module.to(torch.device('cpu'))
    trainer = Trainer(model, batch_size, lr, weight_decay)
    tester = Tester(model, batch_size)

    """Output files."""
    file_CIs= 'output/result/CIs--' + setting + '.txt'
    file_model = 'output/model/' + setting
    CIs = ('Epoch\tTime(sec)\tLoss_train\t'
            'CI\tMSE\trm')
    with open(file_CIs, 'w') as f:
        f.write(CIs + '\n')

    """Start training."""
    print('Training...')
    print(CIs)
    start = timeit.default_timer()
    CI1 = 0
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(train_set, p_LMs, p_SPDs, d_LMs, epoch)
        CI, MSE, rm = tester.test(val_set, p_LMs, p_SPDs, d_LMs)

        end = timeit.default_timer()
        time = end - start

        CIs = [epoch, time, loss_train,
                CI, MSE, rm]
        tester.save_AUCs(CIs, file_CIs)
        # tester.save_model(model, file_model)
        print('\t'.join(map(str, CIs)))
        if CI1 < CI:
            CI1 = CI
            tester.save_model(model, file_model)
