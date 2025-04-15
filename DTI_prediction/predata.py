import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from graph_features import atom_features
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import pickle
import torch
import torch.nn as nn
from Representation import Representation_model



BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25 }

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

num_atom_feat = 34

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency

def first_sequence(sequence):
    words = [protein_dict[sequence[i]]
             for i in range(len(sequence))]
    return np.array(words)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
# device = torch.device('cpu')
# 
# prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
# prot_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)

# chem_tokenizer = AutoTokenizer.from_pretrained("PubChem10M_SMILES_BPE_450k", do_lower_case=False)
# chem_model = AutoModel.from_pretrained("PubChem10M_SMILES_BPE_450k").to(device)

model = Representation_model(3, 128, 256, 64, 1)
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
model_state_dict = torch.load("pretrain_model/SPD_model/ssl_LM")
model.load_state_dict(model_state_dict)

def SPD_feature(LM_feature):
    SPD_features = model.module.LM_generate(LM_feature.to(device).unsqueeze(0)).squeeze(0)
    return SPD_features.cpu()


def DTI_datasets(dataset, dir_input, LM_dic):

    with open(LM_dic, 'rb') as p:
        pro_LM = pickle.load(p)

    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')

    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_SPD, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 1200:
            sequences = sequences[0:1200]
        sequencess.append(sequences)
        smiless.append(smiles)

        if sequences not in p_SPD:
            print(SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32)).shape)
            p_SPD[sequences] = SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32))

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM_SPD.pkl", "wb") as p:
        pickle.dump(p_SPD, p)

    # np.save(dir_input + 'molecule_words', molecule_words)
    # np.save(dir_input + 'molecule_atoms', molecule_atoms)
    # np.save(dir_input + 'molecule_adjs', molecule_adjs)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'sequences', sequencess)
    # np.save(dir_input + 'smiles', smiless)
    # np.save(dir_input + 'interactions', interactions)

def DTI_drugbank_datasets(dataset, dir_input, LM_dic):

    with open(LM_dic, 'rb') as p:
        pro_LM = pickle.load(p)

    with open(dataset, "r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_SPD, d_LM = {}, {}

    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 1200:
            sequences = sequences[0:1200]
        sequencess.append(sequences)
        smiless.append(smiles)

        if sequences not in p_SPD:
            print(SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32)).shape)
            p_SPD[sequences] = SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32))

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM_SPD.pkl", "wb") as p:
        pickle.dump(p_SPD, p)
    # np.save(dir_input + 'molecule_words', molecule_words)
    # np.save(dir_input + 'molecule_atoms', molecule_atoms)
    # np.save(dir_input + 'molecule_adjs', molecule_adjs)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'sequences', sequencess)
    # np.save(dir_input + 'smiles', smiless)
    # np.save(dir_input + 'interactions', interactions)

def DTA_datasets(dataset, dir_input, LM_dic):
    with open(LM_dic, 'rb') as p:
        pro_LM = pickle.load(p)

    with open(dataset, "r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, affinities = [], [], [], [], [], [], []
    p_SPD, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, affinity = data.strip().split(" ")
        if len(sequences) > 1200:
            sequences = sequences[0:1200]
        sequencess.append(sequences)
        smiless.append(smiles)

        if sequences not in p_SPD:
            print(SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32)).shape)
            p_SPD[sequences] = SPD_feature(torch.tensor(pro_LM[sequences], dtype=torch.float32))

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        affinities.append(np.array([float(affinity)]))

    with open(dir_input + "p_LM_SPD.pkl", "wb") as p:
        pickle.dump(p_SPD, p)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'affinities', affinities)

if __name__ == "__main__":
    DTI_datasets("datasets/Human/original/data.txt", 'datasets/Human/data_split/', 'datasets/Human/p_LM.pkl')

    print('The preprocess of dataset has finished!')
