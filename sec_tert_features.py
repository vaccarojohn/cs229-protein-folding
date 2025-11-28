import os
import requests
import tempfile
import numpy as np
import pandas as pd
from biotite.structure.io import load_structure
from biotite.structure.celllist import CellList
from biotite.structure import get_residue_positions

def get_alpha_carbon_list(seq):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=seq)
    
    alpha_carbons = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, 'temp.pdb')

        with open(temp_file_path, 'w') as f:
            f.write(response.content.decode('utf-8'))

        atom_array = load_structure(temp_file_path)
        alpha_carbons = atom_array[atom_array.atom_name == 'CA']

    return alpha_carbons

def main():
    df = pd.read_csv('data_with_subsequence_data.csv')
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'

    for index, row in df.iterrows():
        if not normal_mask[index]:
            df.loc[index, 'CO'] = 0
            df.loc[index, 'Abs_CO'] = 0
            df.loc[index, 'TCD'] = 0
            df.loc[index, 'LR_CO'] = 0
            continue

        alpha_carbons = get_alpha_carbon_list(row['Sub-Sequence Used In Experiment'])
        cell_list = CellList(alpha_carbons, cell_size=5)
        contact_atom_indices_six = cell_list.get_atoms(alpha_carbons.coord, radius=6)
        contact_atom_indices_eight = cell_list.get_atoms(alpha_carbons.coord, radius=8)

        l = len(alpha_carbons)
        sep_co = 0
        sep_tcd = 0
        sep_lrco = 0
        n_co = 0
        n_lrco = 0

        for i in range(contact_atom_indices_six.shape[0]):
            atom_mask = contact_atom_indices_six[i] != -1
            delta_s = np.abs(get_residue_positions(alpha_carbons, contact_atom_indices_six[i][atom_mask]) - i)

            sep_co += np.sum(delta_s)
            n_co += (len(delta_s) - 1)

            far_mask = delta_s >= 2
            sep_tcd += np.sum(delta_s[far_mask])

        for i in range(contact_atom_indices_eight.shape[0]):
            atom_mask = contact_atom_indices_eight[i] != -1
            delta_s = np.abs(get_residue_positions(alpha_carbons, contact_atom_indices_eight[i][atom_mask]) - i)

            far_mask = delta_s >= 12
            sep_lrco += np.sum(delta_s[far_mask])
            n_lrco += np.sum(far_mask)

        if n_co == 0:
            df.loc[index, 'CO'] = 0
            df.loc[index, 'Abs_CO'] = 0
        else:
            df.loc[index, 'CO'] = round(sep_co / (n_co * l), 4)
            df.loc[index, 'Abs_CO'] = round(sep_co / n_co, 4)

        df.loc[index, 'TCD'] = sep_tcd / (l * l)
        
        if n_lrco == 0:
            df.loc[index, 'LR_CO'] = 0
        else:
            df.loc[index, 'LR_CO'] = round(sep_lrco / (n_lrco * l), 4)

    df.to_csv('data_with_tertiary_structure.csv', index=False)

if __name__ == "__main__":
    main()
