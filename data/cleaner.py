import numpy as np
import pandas as pd
import re

#todo: length of clean sequence
#todo: column for 
#1-indexed

csv_name = "Esmée's Cleaned_Aggregated Dataset With Some Missing Features - Sheet1"

df = pd.read_csv(f'{csv_name}.csv')

# Adding new columns to dataframe
df['Sub-Sequence Used In Experiment'] = df['PDB Sequence']
df['Length of Sub-Sequence Used In Experiment'] = len(df['PDB Sequence'])

for i in range(10, 11):
    sequence = df.at[i,'PDB Sequence']
    print(f"sequence: {sequence}")
    if sequence != 'WEIRD':
        #skip NaNs
        if pd.isna(df.at[i, 'PDB']):
            print("oh no!: "+str(pd.isna(df.at[i, 'PDB'])))
            continue

        pdb_code = str(df.at[i, 'PDB'])
        print(f"pdb code: {pdb_code}")

        #normalize dashes
        for dash in ['–', '—', '−', '‒', '―']:
            pdb_code = pdb_code.replace(dash, '-')

        # use the LAST (...) pair if there are multiple
        start = pdb_code.rfind('(')
        end   = pdb_code.find(')', start + 1)

        inside = None
        if start != -1 and end != -1:
            inside = pdb_code[start + 1:end]

        # extract numbers even with extra text (e.g., "Chain B: 29-335")
        if inside:
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

            if start_idx < end_excl:  # guard against bad ranges (not sure if this is actually necessary, but doesn't hurt)
                df.at[i, 'Sub-Sequence Used In Experiment'] = sequence[start_idx:end_excl]
                df.at[i, 'Length of Sub-Sequence Used In Experiment'] = len(df.at[i, 'Sub-Sequence Used In Experiment'])

df.to_csv(f'{csv_name}_with_subsequence_data.csv', index=False)
