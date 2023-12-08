import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import os
import pickle

def perform_grid_search(X, y):
    parameter_grid = {
        'hidden_layer_sizes': [(3), (5), (7), (9), (8), (3,5), (5,3), (7,5), (7,7), (9,7), (9,9), (8,8), (3,5,3), (5,3,5), (7,5,7), (7,7,7), (9,7,9), (9,9,9), (8,8,8)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'batch_size': [2, 3, 4, 5],
        'max_iter': [2000, 2500, 3000],
    }

    mlp = MLPClassifier(max_iter=1000)
    scorer = make_scorer(accuracy_score)
    grid_search = GridSearchCV(estimator=mlp, param_grid=parameter_grid, scoring=scorer, cv=3)
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    data_dir = os.path.join(project_dir, 'data', 'combined_features', 'combined_data.npy')
    data = np.load(data_dir)
    X = data[:, 4:]  # Features
    y = np.argmax(data[:, :4], axis=1)  # Labels

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Run grid search
    best_params, best_score = perform_grid_search(X_train, y_train)

    # Print the best parameters and the score of the best classifier
    print("Best parameters found:", best_params)
    print("Best scoring accuracy:", best_score)

    # Save the best parameters to a file
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

# Run the main function
if __name__ == "__main__":
    main()
