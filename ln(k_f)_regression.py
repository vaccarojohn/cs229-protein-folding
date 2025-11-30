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

def read_data(filename, tokenizer, max_length):
    df = pd.read_csv(filename)
    df = df[df['Length of Sub-Sequence Used In Experiment'] <= max_length]
    df = df[df['ln(kf) (25°C)'] != 'WEIRD']
    df = df[df['Sub-Sequence Used In Experiment'] != 'WEIRD']
    tokenizer.fit_on_texts(df['Sub-Sequence Used In Experiment'])
    sequences = tokenizer.texts_to_sequences(df['Sub-Sequence Used In Experiment'].tolist())
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating="post", padding="post")
    df['Sub-Sequence Used In Experiment'] = list(padded_sequences)
    return df

def build_model(x_train, x_sequence, y_train, tokenizer, max_length, embed_dim):
    vocab_size = len(tokenizer.word_index) + 1 #plus 1 because there is a padding token
    seq_input = keras.Input(shape=(max_length,), dtype='int32', name='text_input')

    text_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(seq_input)
    lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(text_embedding)
    lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(128))(lstm_output)

    output = keras.layers.Dense(1, activation='linear', name='output')(lstm_output)

    model = keras.Model(inputs=seq_input, outputs=output)
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), metrics=[])
    model.fit(x_sequence, y_train, epochs=30) #model.fit({'text_input': train_pad, 'numerical_input': x_length_train}, y_train, epochs=100)

    return model

def main():
    #Adjustable Hyperparameters
    folds = 5
    max_length = 307
    embedding_dimension = 10
    non_sequence_features = ['Length of Sub-Sequence Used In Experiment']
    label = 'ln(kf) (25°C)'

    #reading in data
    tokenizer = Tokenizer(char_level=True)
    data = read_data('data.csv', tokenizer, max_length)
    

    #formatting data into features and labels
    x = data[non_sequence_features].to_numpy()
    x_sequence = np.stack(data["Sub-Sequence Used In Experiment"].to_numpy()).astype('int32')
    y = data[label].to_numpy()

    kf = KFold(n_splits=folds, shuffle=True)

    final_average_loss = 0
    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        x_train_sequence = x_sequence[train_index]
        x_test = x[test_index]
        x_test_sequence = x_sequence[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_model(x_train, x_train_sequence, y_train, tokenizer, max_length, embedding_dimension)
    
        loss = model.evaluate(x_test_sequence, y_test) #TODO: change this to more features once you start learning with both features

        predictions = model.predict(x_test_sequence) #TODO: change this to more features once you start learning with both features
        actual =  y_test
        inputs = x_test #change to x_train[x.columns.get_loc('Length of Sub-Sequence Used In Experiment')] when you have more features

        #Making the figure
        fig, ax = plt.subplots()
        ax.scatter(inputs, predictions, label='Predicted Folding Rate', color='blue', marker='o')
        ax.scatter(inputs, actual, label='Actual Folding Rate', color='green', marker='x')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Folding Rate')
        ax.legend()
        plt.show()

        print(f"Test Loss: {loss:.4f}")
        final_average_loss += loss
    print(f"Final Average Loss: {final_average_loss/kf.get_n_splits()}")

if __name__ == "__main__":
    main()
