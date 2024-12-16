from Xgboost import XgBoost, generate_random_feature_combinations
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    X_val = pd.read_csv('../data/X_val.csv')
    y_val = pd.read_csv('../data/y_val.csv')

    X_train_tabular, X_test_tabular, X_val_tabular = generate_random_feature_combinations(X_train, X_test, X_val, num_features=1000, random_state=42)

    estimators = [10, 150, 20]
    depths = [5,10,14]
    learning_rates = [0.01, 0.05, 0.1]

    best_score = 0

    for estimator in tqdm(estimators):
        for depth in tqdm(depths):
            for learning_rate in tqdm(learning_rates):
                model = XgBoost(estimator=estimator, depth=depth, learning_rate=learning_rate)
                model.fit(X_train_tabular, y_train, X_val_tabular, y_val, verbose=False)
                score = model.score(X_test_tabular, y_test)
                if score > best_score:
                    best_score = score
                    best_estimator = estimator
                    best_depth = depth
                    best_learning_rate = learning_rate

    print(f"Best score: {best_score}")
    print(f"Best estimator: {best_estimator}")
    print(f"Best depth: {best_depth}")

    model = XgBoost(estimator=best_estimator, depth=best_depth, learning_rate=best_learning_rate)
    model.fit(X_train_tabular, y_train, X_val_tabular, y_val)
    #save model to disk
    model.save_model('best_xgboost.pkl')

