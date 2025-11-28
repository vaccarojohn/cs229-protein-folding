import os
from matplotlib.pylab import f
import requests
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
from biotite.structure.io import load_structure
from biotite.structure.celllist import CellList
from biotite.structure import get_residue_positions
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

XGB_PARAMS = {
    'eta': 0.1,
    'max_depth': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 10,
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

NUM_TREES = 15

def read_data(base_dir, filename):
    df = pd.read_csv(base_dir + "/" + filename)

    two_state_mask = (df['Folding Type'] == '2S') | (df['Folding Type'] == '2S*')
    multistate_mask = df['Folding Type'] == 'N2S'
    normal_mask = df['Sub-Sequence Used In Experiment'] != 'WEIRD'

    features = ['Sub-Sequence Used In Experiment', 'Length of Sub-Sequence Used In Experiment', 'ln(kf) 25', 'CO', 'Abs_CO', 'TCD', 'LR_CO']
    two_state_features = df[two_state_mask & normal_mask][features]
    multistate_features = df[multistate_mask & normal_mask][features]

    two_state_features['State'] = 0
    multistate_features['State'] = 1
    
    return pd.concat([two_state_features, multistate_features])

def plot_xgboost_data(y_score, y_test):
    # plot 2S and N2S separately
    plt.figure()
    plt.hist(y_score[y_test.astype(bool)], bins=np.linspace(0,1,50), histtype='step', color='midnightblue',label='2S')
    plt.hist(y_score[~(y_test.astype(bool))], bins=np.linspace(0,1,50), histtype='step', color='firebrick', label='N2S')

    # make the plot readable
    plt.xlabel('Prediction from BDT', fontsize=12)
    plt.ylabel('Events', fontsize=12)
    plt.legend(frameon=False)

    plt.show()

def build_features(data):
    feature_names = ['len', 'F', 'G', 'C', 'H', 'L', 'R', 'CO', 'Abs_CO', 'TCD', 'LR_CO']
    labels = data['State'].to_numpy()
    
    # Extract primary structure features
    length_data = data['Length of Sub-Sequence Used In Experiment'].to_numpy()
    subseq_data = data['Sub-Sequence Used In Experiment'].to_numpy().astype(str)

    f_data = np.char.count(subseq_data, sub='F') / length_data
    g_data = np.char.count(subseq_data, sub='G') / length_data
    c_data = np.char.count(subseq_data, sub='C') / length_data
    h_data = np.char.count(subseq_data, sub='H') / length_data
    l_data = np.char.count(subseq_data, sub='L') / length_data
    r_data = np.char.count(subseq_data, sub='R') / length_data
    
    # Extract tertiary structure features
    co_data = data['CO'].to_numpy()
    abs_co_data = data['Abs_CO'].to_numpy()
    tcd_data = data['TCD'].to_numpy()
    lr_co_data = data['LR_CO'].to_numpy()

    overall_data = np.column_stack((length_data, f_data, g_data, c_data, h_data, l_data, r_data, co_data, abs_co_data, tcd_data, lr_co_data))
    return (feature_names, overall_data, labels)

def main():
    data = read_data(os.path.normpath(os.path.join(os.path.realpath(__file__), '../../../')), 'data_with_tertiary_structure.csv')

    kf = KFold(n_splits=len(data), shuffle=True)
    param = list(XGB_PARAMS.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    feature_names, x, y = build_features(data)
    final_xgb_accuracy = 0
    final_rf_accuracy = 0
    for train_index, test_index in kf.split(data):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dm_train = xgb.DMatrix(data=x_train, label=y_train, feature_names=feature_names)
        booster = xgb.train(param, dm_train, num_boost_round=NUM_TREES)

        dm_test = xgb.DMatrix(data=x_test, label=y_test, feature_names=feature_names)
        y_score_xgb = booster.predict(dm_test)
        y_pred_xgb = y_score_xgb >= 0.5
        xgb_accuracy = np.sum(y_pred_xgb == y_test) / len(y_test)
        #print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")
        #plot_xgboost_data(y_score_xgb, y_test)
        final_xgb_accuracy += xgb_accuracy / len(data)

        rf_model = RandomForestClassifier(n_estimators=20, max_depth=2, max_features=4)
        rf_model.fit(x_train, y_train)

        y_pred_rf = rf_model.predict(x_test)
        rf_accuracy = np.sum(y_pred_rf == y_test) / len(y_test)
        #print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
        final_rf_accuracy += rf_accuracy / len(data)

    print(f"Final XGBoost Accuracy: {final_xgb_accuracy:.2f}")
    print(f"Final Random Forest Accuracy: {final_rf_accuracy:.2f}")

if __name__ == "__main__":
    main()