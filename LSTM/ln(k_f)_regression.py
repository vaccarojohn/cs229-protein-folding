import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, KFold

####Adjustable Hyperparameters#####

#Training
FOLDS = 5
MAX_LENGTH = 307
EPOCHS = 30

#Model Complexity
SEQUENCE_EMBEDDING_DIMENSION = 20
CATEGORY_EMBEDDING_DIMENSION = 5
NUM_LSTM_NEURONS = 128
NUM_LSTM_LAYERS = 1
NUM_MLP_NEURONS = 512
NUM_MLP_LAYERS = 1

#regularization
LSTM_DROPOUT=0.3
LSTM_RECURRENT_DROPOUT=0.3
LSTM_REGULARIZER = keras.regularizers.l2(1e-4)
MLP_DROPOUT = 0.3
MLP_REGULARIZER = keras.regularizers.l2(1e-4)

#features
CATEGORY_FEATURES = ['Class', 'Fold']
NUMERIC_FEATURES = ['Length_of_Sub-Sequence_Used_In_Experiment', 'B_factor', "ln_ku_25", "CO", "Abs_CO", "TCD", "LR_CO", 'Proportion_F', 'Proportion_G', 'Proportion_C', 'Proportion_L', 'Proportion_R', 'Folding_Type']
SEQUENCE_FEATURES = ['DSSP', 'Sub-Sequence_Used_In_Experiment']
LABEL = 'ln_kf_25'

####Pre-Processing#####

def read_data(filename, category_features, numeric_features, sequence_features, label, features_to_tokenizers, max_length):
    #data-specific cleanup
    df = pd.read_csv(filename)
    all_cols = category_features+numeric_features+sequence_features+[label]
    df = df[all_cols] #keep only the features we'll use
   
    #filter features
    mask = df["Length_of_Sub-Sequence_Used_In_Experiment"] <= max_length
    for feature in all_cols:
        mask &= df[feature]!= 'WEIRD'
    mask &= df[all_cols].notna().all(axis=1)
    df = df.loc[mask, all_cols].reset_index(drop=True)
    
    #tokenizing category features
    for feature in category_features:
        features_to_tokenizers[feature].fit_on_texts(df[feature])
        sequences = features_to_tokenizers[feature].texts_to_sequences(df[feature])
        padded_category_feature = pad_sequences(sequences, maxlen=1, padding="post", truncating="post", dtype="int32", value=0)
        df[feature] = padded_category_feature


    #normalizing numeric features 
    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature])
        df[feature] = (df[feature] - df[feature].mean()) / (df[feature].std(ddof=0) + 1e-8)


    #processing sequential features (tokenizing and padding them)
    for feature in sequence_features:
        features_to_tokenizers[feature].fit_on_texts(df[feature])
        sequences = features_to_tokenizers[feature].texts_to_sequences(df[feature].tolist())
        padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating="post", padding="post", dtype="int32",)
        df[feature] = list(padded_sequences)  #stores each sequence in a single cell, avoiding weird pandas broadcasting stuff
    return df

####Building the Model#####

def build_model(x_train, category_features, numeric_features, sequence_features, y_train, features_to_tokenizers, category_embed_dim, sequence_embed_dim, epochs, max_length):

    feature_to_input_layer = {feat: keras.Input(shape=(1,), dtype='int32', name=feat) for feat in category_features} | {feat: keras.Input(shape=(1,), dtype='float32', name=feat) for feat in numeric_features} | {feat: keras.Input(shape=(max_length,), dtype='int32', name=feat) for feat in sequence_features} 
    
    sequential_feature_to_embedding_layer =  {feat: keras.layers.Embedding(input_dim=len(features_to_tokenizers[feat].word_index)+1, output_dim=sequence_embed_dim, mask_zero=True)(feature_to_input_layer[feat]) for feat in sequence_features}
    merged_sequential_features = keras.layers.Concatenate(axis=2)(list(sequential_feature_to_embedding_layer.values()))

    #We create LSTM according to global variables above
    shared_mask = keras.layers.Lambda(lambda t: tf.not_equal(t, 0))(feature_to_input_layer[sequence_features[0]]) #Masking the sequence with the mask for one of the sequential features (since they should all be the same)
    if NUM_LSTM_LAYERS == 1:
        lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(NUM_LSTM_NEURONS, return_sequences=False, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT, kernel_regularizer=LSTM_REGULARIZER))(merged_sequential_features, mask = shared_mask)
    else:
        for i in range(1, NUM_LSTM_LAYERS+1):
            if  i == 1:
                lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(int(NUM_LSTM_NEURONS/i), return_sequences=True, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT, kernel_regularizer=LSTM_REGULARIZER))(merged_sequential_features, mask = shared_mask)
            elif i == NUM_LSTM_LAYERS :
                lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(int(NUM_LSTM_NEURONS/i), return_sequences=False, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT, kernel_regularizer=LSTM_REGULARIZER))(lstm_output, mask = shared_mask)
            else:
                lstm_output = keras.layers.Bidirectional(keras.layers.LSTM(int(NUM_LSTM_NEURONS/i), return_sequences=True, dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT, kernel_regularizer=LSTM_REGULARIZER))(lstm_output, mask = shared_mask)

    category_feature_to_embedding_layer = {feat: keras.layers.Flatten()(keras.layers.Embedding(input_dim=len(features_to_tokenizers[feat].word_index)+1, output_dim=category_embed_dim)(feature_to_input_layer[feat])) for feat in category_features}
   
    merged_lstm_output_and_features = keras.layers.Concatenate(axis=1)([lstm_output] + [feature_to_input_layer[feat] for feat in numeric_features] + [category_feature_to_embedding_layer[feat] for feat in category_features])

    #We create MLP according to global variables above
    if NUM_MLP_LAYERS == 1:
        mlp_output = keras.layers.Dense(NUM_MLP_NEURONS, activation="relu", kernel_regularizer=MLP_REGULARIZER)(merged_lstm_output_and_features)
        mlp_output = keras.layers.Dropout(MLP_DROPOUT)(mlp_output)
    else:
        for i in range(1, NUM_MLP_LAYERS+1):
            if  i == 1:
                mlp_output = keras.layers.Dense(int(NUM_MLP_NEURONS/i), activation="relu", kernel_regularizer=MLP_REGULARIZER)(merged_lstm_output_and_features)
                mlp_output = keras.layers.Dropout(MLP_DROPOUT)(mlp_output)
            elif i == NUM_MLP_LAYERS:
                mlp_output = keras.layers.Dense(int(NUM_MLP_NEURONS/i), activation="relu", kernel_regularizer=MLP_REGULARIZER)(mlp_output)
                mlp_output = keras.layers.Dropout(MLP_DROPOUT)(mlp_output)
            else:
               mlp_output = keras.layers.Dense(int(NUM_MLP_NEURONS/i), activation="relu", kernel_regularizer=MLP_REGULARIZER)(mlp_output)
               mlp_output = keras.layers.Dropout(MLP_DROPOUT)(mlp_output)
    
    output = keras.layers.Dense(1, activation='linear', name='output')(mlp_output)

    model = keras.Model(inputs=feature_to_input_layer, outputs=output)
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(clipnorm=1.0), metrics=[])
    model.fit(x_train, y_train, epochs=epochs)

    return model 

####Training, Testing, and Plotting Data#####

def main():
    #setting up data
    features_to_tokenizers = {feat: Tokenizer(char_level=True) for feat in SEQUENCE_FEATURES} | {feat: Tokenizer(split=None, filters='', lower=False) for feat in CATEGORY_FEATURES}
    data = read_data("../data/lstm_data.csv", CATEGORY_FEATURES, NUMERIC_FEATURES, SEQUENCE_FEATURES, LABEL, features_to_tokenizers, MAX_LENGTH)

    #Train/Test K-Fold Loop
    final_average_loss = 0
    kf = KFold(n_splits=FOLDS, shuffle=True)
    for train_index, test_index in kf.split(data):
        # Clear previous model's state
        keras.backend.clear_session()

        train_fold = data.iloc[train_index]
        x_train = {feat: train_fold[feat].to_numpy().astype("int32").reshape(-1, 1) for feat in CATEGORY_FEATURES} | {feat: train_fold[feat].to_numpy().astype("float32").reshape(-1, 1) for feat in NUMERIC_FEATURES} | {feat: np.stack(train_fold[feat].to_numpy()).astype("int32") for feat in SEQUENCE_FEATURES}
        y_train =  np.asarray(train_fold[LABEL], dtype="float32").reshape(-1, 1)
        
        test_fold = data.iloc[test_index]
        x_test =  {feat: test_fold[feat].to_numpy().astype("int32").reshape(-1, 1) for feat in CATEGORY_FEATURES} | {feat: test_fold[feat].to_numpy().astype("float32").reshape(-1, 1) for feat in NUMERIC_FEATURES} | {feat: np.stack(test_fold[feat].to_numpy()).astype("int32") for feat in SEQUENCE_FEATURES}
        y_test = test_fold[LABEL].to_numpy().astype('float32').reshape(-1, 1)                 # should be (89, 1)
        model = build_model(x_train, CATEGORY_FEATURES, NUMERIC_FEATURES, SEQUENCE_FEATURES, y_train, features_to_tokenizers, CATEGORY_EMBEDDING_DIMENSION, SEQUENCE_EMBEDDING_DIMENSION, EPOCHS, MAX_LENGTH)

        loss = model.evaluate(x_test, y_test) 
        predictions = model.predict(x_test) 
        lengths = test_fold['Length_of_Sub-Sequence_Used_In_Experiment'].to_numpy(dtype="float32")
        actual =  y_test

        #Making the figure for the current fold
        fig, ax = plt.subplots()
        
        ax.scatter(lengths, predictions, label='Predicted Folding Rate', color='blue', marker='o')
        ax.scatter(lengths, actual, label='Actual Folding Rate', color='green', marker='x')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Folding Rate')
        ax.legend()
        plt.show()

        print(f"Test Loss: {loss:.4f}")
        final_average_loss += loss
    print(f"Final Average Loss: {final_average_loss/kf.get_n_splits()}")

if __name__ == "__main__":
    main()
