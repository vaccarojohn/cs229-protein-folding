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

def read_data(base_dir, filename):
    df = pd.read_csv(base_dir + "/" + filename)

    two_state_mask = (df['Folding Type'] == '2S') | (df['Folding Type'] == '2S*')
    multistate_mask = df['Folding Type'] == 'N2S'
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'
    length_mask = df['Length of Sub-Sequence Used In Experiment'] <= 141

    features = ['Sub-Sequence Used In Experiment', 'Length of Sub-Sequence Used In Experiment']
    two_state_features = df[two_state_mask & normal_mask & length_mask][features]
    multistate_features = df[multistate_mask & normal_mask & length_mask][features]

    two_state_features['State'] = 0
    multistate_features['State'] = 1
    
    return pd.concat([two_state_features, multistate_features])

def build_model(x_text_train, x_length_train, y_train, tokenizer):
    train_seq = tokenizer.texts_to_sequences(x_text_train.tolist())
    train_pad = pad_sequences(train_seq, maxlen=141, truncating="post", padding="post")
        
    text_input = keras.Input(shape=(141, 1,), dtype='int32', name='text_input')
    rnn_output = keras.layers.SimpleRNN(64)(text_input)
    numerical_input = keras.Input(shape=(1, ), dtype='int32', name='numerical_input')

    combined_processed = keras.layers.Dense(64, activation='sigmoid', name='processing')(keras.layers.concatenate([rnn_output, numerical_input]))
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(combined_processed)
    model = keras.Model(inputs=[text_input, numerical_input], outputs=output)
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit({'text_input': train_pad, 'numerical_input': x_length_train}, y_train, epochs=100)
    return model

def main():
    data = read_data(os.path.normpath(os.path.join(os.path.realpath(__file__), '../../../')), 'data.csv')

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(data['Sub-Sequence Used In Experiment'])

    kf = KFold(n_splits=5, shuffle=True)

    x_text_data = data[['Sub-Sequence Used In Experiment']].values
    x_length_data = data[['Length of Sub-Sequence Used In Experiment']].values
    y_data = data['State'].values

    for train_index, test_index in kf.split(x_length_data):
        x_text_train, x_length_train = x_text_data[train_index], x_length_data[train_index]
        x_text_test, x_length_test = x_text_data[test_index], x_length_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        model = build_model(x_text_train, x_length_train, y_train, tokenizer)

        test_seq = tokenizer.texts_to_sequences(x_text_test.tolist())
        test_pad = pad_sequences(test_seq, maxlen=141, truncating="post", padding="post")
    
        loss, accuracy = model.evaluate({'text_input': test_pad, 'numerical_input': x_length_test}, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()