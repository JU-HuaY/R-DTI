from transformers import AutoModel, AutoTokenizer
from transformers import T5Tokenizer, T5EncoderModel
# from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from Bio.PDB import PDBParser
import pickle
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")

aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
prot_tokenizer = T5Tokenizer.from_pretrained('pre_trained_model/prot_t5_xl_uniref50', do_lower_case=False)
prot_model = T5EncoderModel.from_pretrained("pre_trained_model/prot_t5_xl_uniref50").to(device)
# prot_tokenizer = AutoTokenizer.from_pretrained("pre_trained_model/prot_bert_bfd", do_lower_case=False)
# prot_model = AutoModel.from_pretrained("pre_trained_model/prot_bert_bfd").to(device)
prot_model.to(torch.float32)

def sequence_feature(sequences):
    protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)
    p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
    p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
    with torch.no_grad():
        prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
    prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
    return prot_feature

def struc_res_coo(path):
    parser = PDBParser()
    structure = parser.get_structure('protein', path)
    res_seq = ''
    res_coos = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # 跳过非标准残基
                if residue.get_resname() == 'UNK':
                    continue
                if residue.get_resname() == 'HOH':
                    continue
                res_id = residue.get_full_id()
                value = aa_codes.setdefault(residue.get_resname(), "X")
                res_seq += value
                res_coo = [0, 0, 0]
                n = 0
                # print(f"Residue {residue.get_resname()}:")
                for atom in residue:
                    # atom_name = atom.get_name()
                    atom_coord = atom.get_coord()
                    res_coo += atom_coord
                    # print(atom_coord)
                    n += 1
                    # print(f"  {atom_name}: {atom_coord}")
                res_coos.append(res_coo / [n, n, n])
    return res_seq, np.array(res_coos)

def data_process(data_pile, out_name):
    res_seq, res_coos = struc_res_coo(data_pile)
    prot_feature = sequence_feature(res_seq)
    data = {
        "Seq": res_seq,
        "Coo": res_coos,
        "LM_f": prot_feature
    }
    with open(out_name + '.pickle', 'wb') as f:
        pickle.dump(data, f)
    # print(data)

# data_process("protein_3d/135l_A_rec.pdb", "1a0g_A")

def main(path_origin, path_output):
    file_pattern = path_origin + '/*.pdb'
    all_pdb_files = glob.glob(file_pattern, recursive=True)
    id = 0
    for file_path in all_pdb_files:
        print(str(id) + ":" + file_path[11:17])
        out_name = path_output + '/' + file_path[11:17]
        data_process(file_path, out_name)
        id += 1


if __name__ == "__main__":
    data_origin = "protein_3d"
    data_out_path = "prot_t5"
    main(data_origin, data_out_path)

