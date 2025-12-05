import numpy as np
import pandas as pd
import re

csv_name = "data"

df = pd.read_csv(f'{csv_name}.csv')

# Adding new columns to dataframe
df['Sub-Sequence Used In Experiment'] = df['PDB Sequence']

for i in range(len(df)):
    sequence = df.at[i,'PDB Sequence']
    print(f"length of seq: {len(sequence)}")
    print(f"sequence: {sequence}")
    if sequence != 'WEIRD':

        pdb_code = str(df.at[i, 'PDB'])
        print(f"pdb code: {pdb_code}")

        #normalize dashes
        for dash in ['–', '—', '−', '‒', '―']:
            pdb_code = pdb_code.replace(dash, '-')

        # use the last pair of parentheses if there are multiple
        start = pdb_code.rfind('(')
        end   = pdb_code.find(')', start + 1)

        # extract what's inside the parentheses
        inside = None
        if start != -1 and end != -1:
            inside = pdb_code[start + 1:end]

        # extract numbers even with extra text (e.g. "Chain B: 29-335")
        if inside:
            print(f"Inside: {inside}")
            nums = re.findall(r'\d+', inside) 
        else:
            nums = []

        if len(nums) >= 2:
            start_1, end_1 = map(int, nums[:2])
            print(f"new pdb code: {pdb_code}")

            # Convert 1-indexed inclusive range to Python slice
            start_idx = start_1 - 1
            end_excl = end_1
            print(f"indecies: {start_idx}, {end_excl}")
        else:
            df.at[i, 'Sub-Sequence Used In Experiment'] = sequence
            df.at[i, 'Length of Sub-Sequence Used In Experiment'] = len(sequence)


df.to_csv(f'{csv_name}_with_subsequence_data.csv', index=False)
