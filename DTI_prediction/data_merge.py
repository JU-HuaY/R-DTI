import numpy as np
import torch
import pickle

def load_tensor(file_name, dtype, device):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def train_data_load(dataset, device, DTI=True):

    molecule_words_train = load_tensor(dataset + '/train/molecule_words', torch.LongTensor, device)
    molecule_atoms_train = load_tensor(dataset + '/train/molecule_atoms', torch.LongTensor, device)
    molecule_adjs_train = load_tensor(dataset + '/train/molecule_adjs', torch.LongTensor, device)
    proteins_train = load_tensor(dataset + '/train/proteins', torch.LongTensor, device)
    sequence_train = np.load(dataset + '/train/sequences.npy')
    smiles_train = np.load(dataset + '/train/smiles.npy')
    if DTI == True:
        interactions_train = load_tensor(dataset + '/train/interactions', torch.LongTensor, device)
    else:
        interactions_train = load_tensor(dataset + '/train/affinities', torch.FloatTensor, device)

    with open(dataset + '/train/p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)

    with open(dataset + '/train/p_LM_SPD.pkl', 'rb') as p:
        p_SPDs = pickle.load(p)

    with open(dataset + '/train/d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)

    return molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, sequence_train, smiles_train, p_LM, p_SPDs, d_LM, interactions_train

def test_data_load(dataset, device, DTI=True):
    molecule_words_test = load_tensor(dataset + '/test/molecule_words', torch.LongTensor, device)
    molecule_atoms_test = load_tensor(dataset + '/test/molecule_atoms', torch.LongTensor, device)
    molecule_adjs_test = load_tensor(dataset + '/test/molecule_adjs', torch.LongTensor, device)
    proteins_test = load_tensor(dataset + '/test/proteins', torch.LongTensor, device)
    sequence_test = np.load(dataset + '/test/sequences.npy')
    smiles_test = np.load(dataset + '/test/smiles.npy')
    if DTI == True:
        interactions_test = load_tensor(dataset + '/test/interactions', torch.LongTensor, device)
    else:
        interactions_test = load_tensor(dataset + '/test/affinities', torch.FloatTensor, device)

    with open(dataset + '/test/p_LM.pkl', 'rb') as p:
        p_LM = pickle.load(p)

    with open(dataset + '/test/p_LM_SPD.pkl', 'rb') as p:
        p_SPDs = pickle.load(p)

    with open(dataset + '/test/d_LM.pkl', 'rb') as d:
        d_LM = pickle.load(d)

    return molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, sequence_test, smiles_test, p_LM, p_SPDs, d_LM, interactions_test

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def Source_ID(ID, length):
    source_label = torch.ones(length) * ID
    return source_label

def merge(train_list, test_list, device, DTI=True):
    molecule_words_trains, molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains = [], [], [], [], [], [], []
    molecule_words_tests, molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests = [], [], [], [], [], [], []
    p_LMs, p_SPDs, d_LMs = {}, {}, {}
    train_source = []
    ID = 0
    for dataset in train_list:
        molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, sequence_train, smiles_train, p_LM, p_SPD, d_LM, interactions_train = train_data_load(dataset, device, DTI)
        molecule_words_trains.extend(molecule_words_train)
        molecule_atoms_trains.extend(molecule_atoms_train)
        molecule_adjs_trains.extend(molecule_adjs_train)
        proteins_trains.extend(proteins_train)
        sequence_trains.extend(sequence_train)
        smiles_trains.extend(smiles_train)
        p_LMs.update(p_LM)
        p_SPDs.update(p_SPD)
        d_LMs.update(d_LM)
        interactions_trains.extend(interactions_train)
        length = len(molecule_words_train)
        train_source.extend(Source_ID(ID, length))
        ID += 1

    for dataset in test_list:
        molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, sequence_test, smiles_test, p_LM, p_SPD, d_LM, interactions_test = test_data_load(dataset, device, DTI)
        molecule_words_tests.extend(molecule_words_test)
        molecule_atoms_tests.extend(molecule_atoms_test)
        molecule_adjs_tests.extend(molecule_adjs_test)
        proteins_tests.extend(proteins_test)
        sequence_tests.extend(sequence_test)
        smiles_tests.extend(smiles_test)
        p_LMs.update(p_LM)
        p_SPDs.update(p_SPD)
        d_LMs.update(d_LM)
        interactions_tests.extend(interactions_test)

    train_dataset = list(zip(molecule_words_trains, molecule_atoms_trains, molecule_adjs_trains, proteins_trains, sequence_trains, smiles_trains, interactions_trains, train_source))
    train_dataset = shuffle_dataset(train_dataset, 1234)

    test_dataset = list(zip(molecule_words_tests, molecule_atoms_tests, molecule_adjs_tests, proteins_tests, sequence_tests, smiles_tests, interactions_tests))
    test_dataset = shuffle_dataset(test_dataset, 1234)

    return train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs

def data_load(data_select, device):
    # p_LMs, d_LMs = {}, {}
    if data_select == "B_to_B":
        train_list = ["datasets/BindingDB"]
        test_list = ["datasets/BindingDB"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge2(train_list, test_list, device)
    elif data_select == "D_to_D":
        train_list = ["datasets/Drugbank"]
        test_list = ["datasets/Drugbank"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "H_to_H":
        train_list = ["datasets/human"]
        test_list = ["datasets/human"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "C_to_C":
        train_list = ["datasets/celegans"]
        test_list = ["datasets/celegans"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device)
    elif data_select == "Da_to_Da":
        train_list = ["datasets/Davis"]
        test_list = ["datasets/Davis"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device, DTI=False)
    elif data_select == "K_to_K":
        train_list = ["datasets/KIBA"]
        test_list = ["datasets/KIBA"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device, DTI=False)
    else:
        train_list = ["datasets/Human"]
        test_list = ["datasets/Human"]
        train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs = merge(train_list, test_list, device)

    return train_dataset, test_dataset, p_LMs, p_SPDs, d_LMs
