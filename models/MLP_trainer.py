import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import os
import itertools

# Function to initialize and train the MLP
def create_mlp(X, y, hidden_layers, activation, alpha, batch_size, learning_rate, learning_rate_init, max_iter, solver, train_test_ratio, output_folder=None):
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_ratio, stratify=y, random_state=1)
    
    # MLP initialization with new parameters
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter, solver=solver, random_state=1)
    
    # Training the MLP
    mlp.fit(X_train, y_train)
    
    # Save the trained MLP to the output folder
    if output_folder:
        with open(os.path.join(output_folder, 'trained_mlp.pkl'), 'wb') as f:
            pickle.dump(mlp, f)
    
    # Generate and save the confusion matrix to the output folder
    y_pred = mlp.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Label names for the confusion matrix
    labels = ['Fantasy', 'Comedy', 'Sci-fi', 'Horror']

    # Plotting the confusion matrix with label names
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Adding the text in each cell
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    if output_folder:
        plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))

    # Calculate feature importance
    feature_importances = np.abs(mlp.coefs_[0]).sum(axis=1)
    top_features_indices = np.argsort(feature_importances)[-10:]  # Indices of top 10 features
    top_features_importances = feature_importances[top_features_indices]  # Importance of top 10 features

    # Generating a report
    report = classification_report(y_test, y_pred, target_names=labels)
    report += f"\n\nTop 10 Feature Importances:\n"
    for index, importance in zip(top_features_indices, top_features_importances):
        report += f"Feature {index}: Importance {importance:.4f}\n"
    report += f"\n\nHidden Layers: {hidden_layers}\n"
    report += f"Activation: {activation}\nAlpha: {alpha}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\nLearning Rate Initialization: {learning_rate_init}\nMax Iterations: {max_iter}\nSolver: {solver}\nTrain-Test Split Ratio: {train_test_ratio}\n"
    
    return mlp, cm, report

# Main function to interact with the user
def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_dir, 'data', 'combined_features', 'combined_data.npy')
    data = np.load(data_dir)
    X = data[:, 4:]  # Features
    y = np.argmax(data[:, :4], axis=1)  # Labels

    # Prompt for user inputs
    hidden_layers_input = input("Enter the number of neurons in each hidden layer separated by commas (e.g., 100,50): ")
    hidden_layers = tuple(map(int, hidden_layers_input.split(',')))

    activation = input("Enter the activation function (e.g., relu, tanh, logistic): ")
    alpha = float(input("Enter the value of alpha (L2 penalty term): "))
    batch_size = int(input("Enter the batch size: "))
    learning_rate = input("Enter the learning rate type (constant, invscaling, adaptive): ")
    learning_rate_init = float(input("Enter the initial learning rate: "))
    max_iter = int(input("Enter the maximum number of iterations: "))
    solver = input("Enter the solver for weight optimization (lbfgs, sgd, adam): ")
    train_test_ratio = float(input("Enter the ratio of training data to total data (e.g., 0.8 for 80% training data): "))

    # Create the new output folder
    output_dir = os.path.join(project_dir, 'outputs')
    output_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    highest_folder_num = max([int(f.split('_')[1]) for f in output_folders]) if output_folders else 0
    new_output_folder = os.path.join(output_dir, f"output_{highest_folder_num + 1}")
    os.makedirs(new_output_folder)

    # Create and train the MLP, and save everything to the output folder
    mlp, cm, report = create_mlp(X, y, hidden_layers, activation, alpha, batch_size, learning_rate, learning_rate_init, max_iter, solver, train_test_ratio, output_folder=new_output_folder)

    # Save the report to the new output folder
    report_file = os.path.join(new_output_folder, 'report.txt')
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_file}")
    print("MLP trained and saved. Confusion matrix and report generated in the output folder.")

# Run the main function
if __name__ == "__main__":
    main()
