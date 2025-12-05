import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=307):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ProteinSequenceTransformerClassifier(nn.Module):
    def __init__(self, seq_size=307, other_features=5, vocab_size=61, d_model=24, nhead=4, num_encoder_layers=1, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.seq_size = seq_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_size) # Custom PositionalEncoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.collapse_embedding = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.collapse_sequence = nn.Linear(seq_size, 1)
        self.final = nn.Linear(1 + other_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, other):
        src = self.embedding(seq) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        transformer_output = self.transformer_encoder(src)
        output = torch.squeeze(self.collapse_embedding(transformer_output))
        output = self.relu(output)
        output = self.collapse_sequence(output)
        final_output = torch.squeeze(self.final(torch.cat((output, other), dim=1)))
        return self.sigmoid(final_output)

AMINO_ACID_VOCABULARY = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}

class AminoAcidTokenizer():
    def __init__(self, max_len=10):
        self.max_len = max_len

    def tokenize(self, seq_list):
        my_array = np.zeros(((len(seq_list), self.max_len)))
        for i, seq in enumerate(seq_list):
            seq_list = list(seq[0])
            sec_list = list(seq[1])
            tokenized = np.array([AMINO_ACID_VOCABULARY[seq_list[i]] * (int(sec_list[i]) + 1) for i in range(len(seq_list))])
            my_array[i, 0 : min(self.max_len, len(tokenized))] = tokenized[0 : min(self.max_len, len(tokenized))]

        return my_array

def read_data(base_dir, filename):
    df = pd.read_csv(base_dir + "/" + filename)

    two_state_mask = (df['Folding Type'] == '2S') | (df['Folding Type'] == '2S*')
    multistate_mask = df['Folding Type'] == 'N2S'
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'

    features = ['Sub-Sequence Used In Experiment', 'DSSP', 'ln(kf) 25', 'Length of Sub-Sequence Used In Experiment', 'CO', 'Abs_CO', 'TCD', 'LR_CO']
    two_state_features = df[two_state_mask & normal_mask][features]
    multistate_features = df[multistate_mask & normal_mask][features]

    two_state_features['State'] = 0
    multistate_features['State'] = 1
    
    return pd.concat([two_state_features, multistate_features])

def main():
    data = read_data(os.path.normpath(os.path.join(os.path.realpath(__file__), '../../../')), 'data_with_sec_tert_structure.csv')

    tokenizer = AminoAcidTokenizer(max_len=307)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    x_text_data = data[['Sub-Sequence Used In Experiment', 'DSSP']].values
    x_other_data = data[['ln(kf) 25', 'CO', 'Abs_CO', 'TCD', 'LR_CO']].values
    y_data = data['State'].values

    #x_length_data = data[['Length of Sub-Sequence Used In Experiment']].values
    #x_lnkf_data = data[['ln(kf) 25']].values
    #plot_protein_lengths(x_length_data, x_lnkf_data, y_data)

    final_transformer_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(x_text_data)):
        x_train, x_test = x_text_data[train_index], x_text_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        scaler = StandardScaler()
        train_seq = torch.from_numpy(tokenizer.tokenize(x_train)).to(torch.int64)
        train_other = torch.from_numpy(scaler.fit_transform(x_other_data[train_index])).to(torch.float32)
        train_labels = torch.from_numpy(y_train).to(torch.int64)

        dataset = TensorDataset(train_seq, train_other, train_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = ProteinSequenceTransformerClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_correct = 0
            for seq, other, tgt in dataloader:
                optimizer.zero_grad()
                output = model(seq, other)
                loss = criterion(output, tgt.to(torch.float32))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                predicted = output >= 0.5
                num_correct += (predicted == tgt).sum().item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.2f}, Accuracy: {num_correct / len(train_labels):.2f}")

        test_seq = torch.from_numpy(tokenizer.tokenize(x_test)).to(torch.int64)
        test_other = torch.from_numpy(scaler.transform(x_other_data[test_index])).to(torch.float32)
        test_labels = torch.from_numpy(y_test).to(torch.int64)
        
        test_dataset = TensorDataset(test_seq, test_other, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32)
        num_correct = 0
        for seq, other, tgt in test_loader:
            output = model(seq, other)
            predicted = output >= 0.5
            num_correct += (predicted == tgt).sum().item()
        
        print(f"Accuracy: {num_correct / len(test_labels):.2f}")
        final_transformer_accuracy += num_correct / len(test_labels)

    print(f"Final Accuracy: {final_transformer_accuracy / 5:.2f}")

if __name__ == "__main__":
    main()