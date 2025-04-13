from Bio.PDB import PDBParser

aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}

def struc_res_coo(path):
    parser = PDBParser()
    structure = parser.get_structure('protein', path)
    res_seq = ''
    res_coos = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == 'UNK':
                    continue
                if residue.get_resname() == 'HOH':
                    continue
                res_id = residue.get_full_id()
                value = aa_codes.setdefault(residue.get_resname(), "X")
                res_seq += value
                res_coo = [0, 0, 0]
                n = 0
                print(f"Residue {residue.get_resname()}:")
                for atom in residue:
                    # atom_name = atom.get_name()
                    atom_coord = atom.get_coord()
                    res_coo += atom_coord
                    print(atom_coord)
                    n += 1
                    # print(f"  {atom_name}: {atom_coord}")
                res_coos.append(res_coo / [n, n, n])
    return res_seq, np.array(res_coos)


struc_res_coo("protein_3d/1a0g_A_rec.pdb")
