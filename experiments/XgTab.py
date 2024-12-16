import torch
import pandas as pd
import pickle as pkl
import numpy as np
import warnings
import xgboost as xgb
import os

from aux import generate_random_feature_combinations, plot_result, plot_error_distribution

np.random.seed(42)

MODEL_PATH = '../models/xg_social.pkl'

# Función para cargar o entrenar el modelo
def load_or_train_model(X_train, y_train):
    if os.path.exists(MODEL_PATH):
        print("Modelo encontrado. Cargando el modelo...")
        with open(MODEL_PATH, 'rb') as f:
            model = pkl.load(f)
    else:
        print("Modelo no encontrado. Entrenando el modelo...")
        model = xgb.XGBRegressor(n_estimators=2000, max_depth=18, learning_rate=0.01)
        model.fit(X_train, y_train.values, verbose=True)
        with open(MODEL_PATH, 'wb') as f:
            pkl.dump(model, f)
    return model




if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Cargar los datos
    X_train_tabular = pd.read_csv('../data/X_train.csv')
    X_test_tabular = pd.read_csv('../data/X_test.csv')
    X_val_tabular = pd.read_csv('../data/X_val.csv')

    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    y_val = pd.read_csv('../data/y_val.csv')

    warnings.filterwarnings("ignore")

    # Cargar o entrenar el modelo
    model = load_or_train_model(X_train_tabular, y_train)

    y_pred_test = model.predict(X_test_tabular)
    y_pred_val = model.predict(X_val_tabular)
    y_pred_train = model.predict(X_train_tabular)

    # Guardar las predicciones como csv
    df = pd.DataFrame(y_pred_train)
    df.to_csv('../data/y_pred_train.csv', header=False)    

