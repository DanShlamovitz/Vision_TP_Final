import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import pickle as pkl
import numpy as np 
import xgboost as xgb
import warnings

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

X_train = pd.read_csv('../X_train.csv')
X_test = pd.read_csv('../X_test.csv')
y_train = pd.read_csv('../y_train.csv')
y_test = pd.read_csv('../y_test.csv')
X_train_resnet = pd.read_csv('../X_train_resnet.csv', header=None).astype(float)
X_test_resnet = pd.read_csv('../X_test_resnet.csv', header=None).astype(float)

X_val = pd.read_csv('../X_val.csv')
y_val = pd.read_csv('../y_val.csv')
X_val_resnet = pd.read_csv('../X_val_resnet.csv', header=None).astype(float)


class XgBoost(nn.Module):
    def __init__(self):
        super(XgBoost, self).__init__()
       # self.xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01, objective='reg:squarederror')+       
        self.xgb = xgb.XGBRegressor(n_estimators=2000, max_depth=5, learning_rate=0.08, objective='reg:squarederror')

import xgboost as xgb
import matplotlib.pyplot as plt

class XgBoost(nn.Module):
    def __init__(self):
        super(XgBoost, self).__init__()
        self.xgb = xgb.XGBRegressor(
            n_estimators=2000, 
            max_depth=5, 
            learning_rate=0.08, 
            objective='reg:squarederror'
        )
    
    def forward(self, x):
        return self.xgb.predict(x)
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True, early_stopping_rounds=50):
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]  # Entrenamiento y validación
            eval_metric = ["rmse"]
            self.xgb.fit(X, y, eval_set=eval_set, eval_metric=eval_metric, 
                         early_stopping_rounds=early_stopping_rounds, verbose=verbose)
            
            # Guardamos los resultados de la pérdida en cada iteración
            results = self.xgb.evals_result()
            self.plot_loss_curve(results)
        else:
            self.xgb.fit(X, y, verbose=verbose)

    def plot_loss_curve(self, results):

        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        # Gráficas de pérdida
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, results['validation_0']['rmse'], label='Train Loss')
        plt.plot(x_axis, results['validation_1']['rmse'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


def generate_random_feature_combinations(X_train, X_test,X_val, num_features, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    X_val_new = X_val.copy()

    columns = X_train.columns
    selected_pairs = []
    
    operations = ['*', '/']
    powers = ['^2', '^3', '^0.5', '^1.5'] 
    
    # Generate random column pairs and operations
    for _ in range(num_features):
        col1, col2 = np.random.choice(columns, 2, replace=False)
        operation = np.random.choice(operations + powers)
        selected_pairs.append((col1, col2, operation))
    
    # Apply the same pairs and operations to both X_train and X_test
    for col1, col2, operation in selected_pairs:
        new_feature_name = f"{col1}_{operation}_{col2}"
        

        if operation == '-':
            X_train_new[new_feature_name] = X_train_new[col1] - X_train_new[col2]
            X_test_new[new_feature_name] = X_test_new[col1] - X_test_new[col2]
            X_val_new[new_feature_name] = X_val_new[col1] - X_val_new[col2]

        elif operation == '/':
            # Avoid division by zero using np.where
            X_train_new[new_feature_name] = np.where(X_train_new[col2] != 0, 
                                                      X_train_new[col1] / X_train_new[col2], 
                                                      np.nan)
            X_test_new[new_feature_name] = np.where(X_test_new[col2] != 0, 
                                                     X_test_new[col1] / X_test_new[col2], 
                                                     np.nan)
            X_val_new[new_feature_name] = np.where(X_val_new[col2] != 0,
                                                    X_val_new[col1] / X_val_new[col2],
                                                    np.nan)
        elif operation == '^2':  # Square of the first column
            X_train_new[new_feature_name] = X_train_new[col1] ** 2
            X_test_new[new_feature_name] = X_test_new[col1] ** 2
            X_val_new[new_feature_name] = X_val_new[col1] ** 2
        elif operation == '^3':  # Cube of the first column
            X_train_new[new_feature_name] = X_train_new[col1] ** 3
            X_test_new[new_feature_name] = X_test_new[col1] ** 3
            X_val_new[new_feature_name] = X_val_new[col1] ** 3
        elif operation == '^0.5':  # Square root of the first column
            X_train_new[new_feature_name] = np.sqrt(X_train_new[col1])
            X_test_new[new_feature_name] = np.sqrt(X_test_new[col1])
            X_val_new[new_feature_name] = np.sqrt(X_val_new[col1])
    
    return X_train_new, X_test_new, X_val_new

# Example usage
warnings.filterwarnings("ignore")
X_train_new, X_test_new, X_val_new = generate_random_feature_combinations(X_train, X_test, X_val, num_features=1000, random_state=42)

print(X_train_new.head())
print(X_test_new.head())

X_train_combined = pd.concat([X_train_resnet, X_train_new], axis=1)
X_test_combined = pd.concat([X_test_resnet, X_test_new], axis=1)
X_val_combined = pd.concat([X_val_resnet, X_val_new], axis=1)


X_train_resnet.columns = [f'col_{i}' for i in range(X_train_resnet.shape[1])]
X_test_resnet.columns = [f'col_{i}' for i in range(X_test_resnet.shape[1])]
X_val_resnet.columns = [f'col_{i}' for i in range(X_val_resnet.shape[1])]



model = XgBoost()
model.fit(X_train_combined, y_train, X_val_combined, y_val, verbose=True)
y_pred2 = model(X_test_combined)

plt.scatter(y_test, y_pred2)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()

#save model
import joblib
joblib.dump(model, 'xgboost_1000est_5depth_combined_data.pkl')

