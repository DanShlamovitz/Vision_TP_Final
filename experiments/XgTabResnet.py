import pandas as pd
import pickle as pkl
import numpy as np 
import warnings
import xgboost as xgb
from sklearn.decomposition import PCA
np.random.seed(42)
import os

from aux import generate_random_feature_combinations, plot_result, plot_predictions_vs_true


def load_or_train_model(X_train, y_train):
    MODEL_PATH = "../models/xg_tabresnetpca.pkl"
    # Verifica si el modelo ya existe
    if os.path.exists(MODEL_PATH):
        print("Modelo encontrado. Cargando el modelo...")
        with open(MODEL_PATH, 'rb') as f:
            model = pkl.load(f)
    else:
        print("Modelo no encontrado. Entrenando el modelo...")
        model = xgb.XGBRegressor(n_estimators=2000, max_depth=18, learning_rate=0.01)
        model.fit(X_train, y_train.values, verbose=True)
        # Guardar el modelo entrenado
        with open(MODEL_PATH, 'wb') as f:
            pkl.dump(model, f)
    return model


if __name__ == '__main__':
    X_train_tabular = pd.read_csv('../data/X_train.csv')
    X_test_tabular = pd.read_csv('../data/X_test.csv')
    X_val_tabular = pd.read_csv('../data/X_val.csv')

    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    y_val = pd.read_csv('../data/y_val.csv')

    X_train_hihglevel = pd.read_csv('../data/X_train_highlevel_features.csv')
    X_test_hihglevel = pd.read_csv('../data/X_test_highlevel_features.csv')
    X_val_hihglevel = pd.read_csv('../data/X_val_highlevel_features.csv')

    
    X_train_resnet = pd.read_csv('../data/train_clip_features.csv', header=None).astype(float)
    X_test_resnet = pd.read_csv('../data/test_clip_features.csv', header=None).astype(float)
    X_val_resnet = pd.read_csv('../data/val_clip_features.csv', header=None).astype(float)

    warnings.filterwarnings("ignore")

    components = 300
    pca = PCA(n_components=components)

    X_train_resnet_pca = pca.fit_transform(X_train_resnet)
    X_test_resnet_pca = pca.transform(X_test_resnet)
    X_val_resnet_pca = pca.transform(X_val_resnet)

    X_train_combined = np.hstack([X_train_tabular, X_train_resnet_pca])
    X_test_combined = np.hstack([X_test_tabular, X_test_resnet_pca])
    X_val_combined = np.hstack([X_val_tabular, X_val_resnet_pca])

    X_train_combined = np.nan_to_num(X_train_combined)
    X_test_combined = np.nan_to_num(X_test_combined)
    X_val_combined = np.nan_to_num(X_val_combined)

    columns = list(X_train_tabular.columns) + [f'PC{i+1}' for i in range(components)]
    X_train_combined_df = pd.DataFrame(X_train_combined, columns=columns)
    X_test_combined_df = pd.DataFrame(X_test_combined, columns=columns)
    X_val_combined_df = pd.DataFrame(X_val_combined, columns=columns)

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    model = load_or_train_model(X_train_combined_df, y_train)
    y_pred = model.predict(X_test_combined_df)

    plot_result(X_test_combined_df, y_test, model)

