import torch
import pandas as pd
import pickle as pkl
import numpy as np
import warnings
import xgboost as xgb
import os  # Importar para verificar la existencia del archivo

from aux import generate_random_feature_combinations, plot_result, plot_predictions_vs_true

np.random.seed(42)

MODEL_PATH = '../models/xg_social_featureeng.pkl'

# Función para cargar o entrenar el modelo
def load_or_train_model(X_train, y_train):
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

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    X_train_tabular = pd.read_csv('../data/X_train.csv')
    X_test_tabular = pd.read_csv('../data/X_test.csv')
    X_val_tabular = pd.read_csv('../data/X_val.csv')

    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    y_val = pd.read_csv('../data/y_val.csv')

    warnings.filterwarnings("ignore")

    X_train_tabular2, X_test_tabular2, X_val_tabular2 = generate_random_feature_combinations(X_train_tabular, X_test_tabular, X_val_tabular, num_features=1000, random_state=42)


    model = load_or_train_model(X_train_tabular2, y_train)
    y_pred = model.predict(X_test_tabular2)

    # plt.plot(y_test, y_pred, 'o')
    # plt.xlabel('y_test')
    # plt.ylabel('y_pred')
    # plt.title('y_test vs y_pred')
    # plt.show()

    
