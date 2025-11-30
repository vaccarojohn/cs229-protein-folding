import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression

def read_data(base_dir, filename):
    df = pd.read_csv(base_dir + "/" + filename)

    two_state_mask = (df['Folding Type'] == '2S') | (df['Folding Type'] == '2S*')
    multistate_mask = df['Folding Type'] == 'N2S'
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'
    length_mask = df['Length of Sub-Sequence Used In Experiment'] <= 141

    features = ['Sub-Sequence Used In Experiment', 'Length of Sub-Sequence Used In Experiment', 'ln(kf) 25']
    two_state_features = df[two_state_mask & normal_mask & length_mask][features]
    multistate_features = df[multistate_mask & normal_mask & length_mask][features]

    two_state_features['State'] = 0
    multistate_features['State'] = 1
    
    return pd.concat([two_state_features, multistate_features])

def build_rnn_model(x_train, y_train, tokenizer, verbose=False):
    train_seq = tokenizer.texts_to_sequences(x_train.tolist())
    train_pad = pad_sequences(train_seq, maxlen=141, truncating="post", padding="post")
        
    text_input = keras.Input(shape=(141, 1,), dtype='int32', name='text_input')
    rnn_output = keras.layers.SimpleRNN(64)(text_input)
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(rnn_output)
    model = keras.Model(inputs=[text_input], outputs=output)

    if verbose:
        model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit({'text_input': train_pad}, y_train, epochs=50, verbose=verbose)
    return model

def plot_protein_lengths(x_length, ln_kf, y):
    colors = np.full(len(y), 'b')
    colors[y == 1] = 'r'

    plt.scatter(x_length, ln_kf, c=colors)
    plt.xlabel('Length of Protein Sequence')
    plt.ylabel('ln(k_F)')
    plt.title('Protein Folding Constant by Length')
    plt.show()

def plot_test_accuracy(x_length, ln_kf, y, y_pred):
    colors = np.full(len(y), 'k')
    colors[y == y_pred] = 'g'

    plt.scatter(x_length, ln_kf, c=colors)
    plt.xlabel('Length of Protein Sequence')
    plt.ylabel('ln(k_F)')
    plt.title('Protein Folding Constant by Length')
    plt.show()

def main():
    data = read_data(os.path.normpath(os.path.join(os.path.realpath(__file__), '../../../')), 'data_with_subsequence_data.csv')

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(data['Sub-Sequence Used In Experiment'])

    kf = KFold(n_splits=len(data), shuffle=True, random_state=42)

    x_data = data[['Sub-Sequence Used In Experiment']].values
    y_data = data['State'].values

    #x_length_data = data[['Length of Sub-Sequence Used In Experiment']].values
    #x_lnkf_data = data[['ln(kf) 25']].values
    #plot_protein_lengths(x_length_data, x_lnkf_data, y_data)

    final_rnn_accuracy = 0
    for i, (train_index, test_index) in enumerate(kf.split(x_data)):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        rnn_model = build_rnn_model(x_train, y_train, tokenizer)

        test_seq = tokenizer.texts_to_sequences(x_test.tolist())
        test_pad = pad_sequences(test_seq, maxlen=141, truncating="post", padding="post")
    
        rnn_loss, rnn_accuracy = rnn_model.evaluate({'text_input': test_pad}, y_test, verbose=False)
        final_rnn_accuracy += rnn_accuracy / len(data)
        #print("Trial #" + str(i + 1))
        #print(f"RNN Test Loss: {rnn_loss:.4f}")
        #print(f"RNN Test Accuracy: {rnn_accuracy:.4f}")

    print(f"Final RNN Accuracy: {final_rnn_accuracy:.2f}")

if __name__ == "__main__":
    main()