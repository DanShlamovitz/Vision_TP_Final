import matplotlib.pyplot as plt
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

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
    
    for _ in range(num_features):
        col1, col2 = np.random.choice(columns, 2, replace=False)
        operation = np.random.choice(operations + powers)
        selected_pairs.append((col1, col2, operation))
    
    for col1, col2, operation in selected_pairs:
        new_feature_name = f"{col1}_{operation}_{col2}"

        if operation == '-':
            X_train_new[new_feature_name] = X_train_new[col1] - X_train_new[col2]
            X_test_new[new_feature_name] = X_test_new[col1] - X_test_new[col2]
            X_val_new[new_feature_name] = X_val_new[col1] - X_val_new[col2]
        elif operation == '/':
            X_train_new[new_feature_name] = np.where(X_train_new[col2] != 0, X_train_new[col1] / X_train_new[col2], np.nan)
            X_test_new[new_feature_name] = np.where(X_test_new[col2] != 0,  X_test_new[col1] / X_test_new[col2], np.nan)
            X_val_new[new_feature_name] = np.where(X_val_new[col2] != 0, X_val_new[col1] / X_val_new[col2], np.nan)
        elif operation == '^2':  
            X_train_new[new_feature_name] = X_train_new[col1] ** 2
            X_test_new[new_feature_name] = X_test_new[col1] ** 2
            X_val_new[new_feature_name] = X_val_new[col1] ** 2
        elif operation == '^3':  
            X_train_new[new_feature_name] = X_train_new[col1] ** 3
            X_test_new[new_feature_name] = X_test_new[col1] ** 3
            X_val_new[new_feature_name] = X_val_new[col1] ** 3
        elif operation == '^0.5': 
            X_train_new[new_feature_name] = np.sqrt(X_train_new[col1])
            X_test_new[new_feature_name] = np.sqrt(X_test_new[col1])
            X_val_new[new_feature_name] = np.sqrt(X_val_new[col1])
    
    return X_train_new, X_test_new, X_val_new


def plot_result(X_test,y_test, model):
    y_pred = model.predict(X_test)
    y_test = y_test["targuet"]
    print(y_pred.shape)
    rsme = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("calucle el RSME")

    sns.set(style="whitegrid") 
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, y_pred, color='royalblue', alpha=0.6, edgecolors='black', s=100, label='Predicciones')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2, label='Línea de referencia (y=x)')

    # Añadir título con el valor de RMSE
    plt.title(f"Predicción vs Real - RMSE: {rsme:.2f}", fontsize=16, weight='bold')

    plt.xlabel('Valores verdaderos', fontsize=14)
    plt.ylabel('Valores predichos', fontsize=14)

    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlim([y_test.min() - 1, y_test.max() + 1])
    plt.ylim([y_test.min() - 1, y_test.max() + 1])

    plt.show()

def plot_feature_distribution(X_train, X_test, X_val):
    plt.figure(figsize=(15, 6))
    for i, feature in enumerate(X_train.columns):
        plt.subplot(3, len(X_train.columns)//3 + 1, i+1)
        sns.histplot(X_train[feature], kde=True, color='blue', label='Train', alpha=0.6)
        sns.histplot(X_test[feature], kde=True, color='green', label='Test', alpha=0.6)
        sns.histplot(X_val[feature], kde=True, color='orange', label='Validation', alpha=0.6)
        plt.legend()
        plt.title(f"Distribución de {feature}")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model):
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10, title='Importancia de las características')
    plt.show()

def plot_predictions_vs_true(model, X_test, y_test):
    # Obtener predicciones
    y_pred = model.predict(X_test)
    
    # Calcular el RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Crear el gráfico
    plt.style.use('seaborn-darkgrid')
    plt.figure(figsize=(10, 8))
    
    # Scatter plot con colores degradados
    errors = np.abs(y_test.values - y_pred)  # Error absoluto
    scatter = plt.scatter(y_test, y_pred, c=errors, cmap='coolwarm', alpha=0.7, edgecolor='k', s=80)
    plt.colorbar(scatter, label='Error absoluto')
    
    # Línea de referencia
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea ideal')
    
    # Personalización del gráfico
    plt.title(f'Predicciones vs. Valores Reales (RMSE: {rmse:.2f})', fontsize=18, fontweight='bold', color='navy')
    plt.xlabel('Valores Reales', fontsize=14)
    plt.ylabel('Predicciones', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(color='gray', linestyle='dotted', linewidth=0.8)
    
    # Ajustar los límites de los ejes
    plt.xlim(y_test.min() * 0.95, y_test.max() * 1.05)
    plt.ylim(y_pred.min() * 0.95, y_pred.max() * 1.05)
    
    plt.show()



def plot_residuals(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Errores de Predicción (Residuals)')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuals')
    plt.show()

def plot_error_distribution(y_pred_test, y_test):

    errors = []
    for pred, true in zip(y_pred_test, y_test['targuet'].values):
        errors.append(pred - true)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    var_error = np.var(errors)

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2)

    plt.title('Distribución de errores en el set de test', fontsize=14, weight='bold')
    plt.xlabel('Error', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)

    plt.legend(title=f'Media: {mean_error:.2f}\Varianza: {var_error:.2f}', loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('white')

    plt.tight_layout()
    plt.show()
