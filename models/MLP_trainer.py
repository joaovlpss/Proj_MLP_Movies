import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import os

# Function to initialize and train the MLP
def create_mlp(X, y, hidden_layers, activation, solver, alpha, learning_rate, learning_rate_init, batch_size, max_iter, train_test_ratio):
    genres = ['Fantasy', 'Comedy', 'Sci-fi', 'Horror']
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_ratio)
    
    # MLP initialization
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver,
                        alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                        batch_size=batch_size, max_iter=max_iter, random_state=1)
    
    # Training the MLP
    mlp.fit(X_train, y_train)
    
    # Save the trained MLP
    with open('trained_mlp.pkl', 'wb') as f:
        pickle.dump(mlp, f)
    
    # Generate and save the confusion matrix
    y_pred = mlp.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(genres))
    plt.xticks(tick_marks, genres, rotation=45)
    plt.yticks(tick_marks, genres)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Generating a report
    report = classification_report(y_test, y_pred, target_names=genres)
    confusion_matrix_text = '\nConfusion Matrix:\n\n' + '\n'.join(['\t'.join(map(str, row)) for row in cm])
    full_report = report + '\n\n' + confusion_matrix_text
    with open('classification_report.txt', 'w') as f:
        f.write(full_report)
    
    return mlp, cm, full_report

# Main function to interact with the user
def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_dir, 'data', 'combined_features', 'combined_data.npy')
    data = np.load(data_dir)
    X = data[:, 4:]  # Features
    y = np.argmax(data[:, :4], axis=1)  # Labels

    # Ask the user to select default hyperparameters or custom ones
    use_defaults = input("Would you like to use the default hyperparameters? (yes/no): ").lower()
    if use_defaults == 'yes':
        # Default hyperparameters
        hidden_layers = (24,)
        activation = 'relu'
        solver = 'adam'
        alpha = 0.001
        learning_rate = 'adaptive'
        learning_rate_init = 0.001
        batch_size = 32
        max_iter = 2000
    else:
        # Custom hyperparameters
        num_layers = int(input("Enter the number of hidden layers: "))
        hidden_layers = tuple(int(input(f"Enter the number of neurons in hidden layer {i+1}: ")) for i in range(num_layers))
        activation = input("Enter the activation function (identity, logistic, tanh, relu): ")
        solver = input("Enter the solver (lbfgs, sgd, adam): ")
        alpha = float(input("Enter the L2 penalty (alpha): "))
        learning_rate = input("Enter the learning rate type (constant, invscaling, adaptive): ")
        learning_rate_init = float(input("Enter the initial learning rate: "))
        batch_size = int(input("Enter the batch size: "))
        max_iter = int(input("Enter the maximum number of iterations: "))

    train_test_ratio = float(input("Enter the ratio of training data to total data (e.g., 0.8 for 80% training data): "))

    # Create and train the MLP with the chosen or default hyperparameters
    mlp, cm, full_report = create_mlp(X, y, hidden_layers, activation, solver, alpha, learning_rate, learning_rate_init, batch_size, max_iter, train_test_ratio)

    print("MLP trained and saved. Confusion matrix and report generated.")

# Run the main function
if __name__ == "__main__":
    main()
