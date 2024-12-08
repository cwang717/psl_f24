# %% (PSL) Coding Assignment 5
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import pandas as pd

# %% Set the seed
np.random.seed(9601)

# %%
if False:
    """
    Part 1: Implement the Pegasos Algorithm
    
    - Use a fixed number of epochs, e.g., T = 20.
    - In each epoch, before going through the dataset, consider randomizing the order of the data points. 
    To achieve this, you should set random seeds for shuffling. For this assignment, the seeds used for
    shuffling do not need to be associated with your UIN
    """
# %%
def pegasos(X, y, lambda_, T):
    # Initialize parameters
    beta = np.zeros(X.shape[1])
    alpha = 0
    t = 0

    for epoch in range(1, T + 1):
        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for i in range(X.shape[0]):
            t += 1
            eta_t = 1 / (t * lambda_)

            if y[i] * (np.dot(X[i], beta) + alpha) < 1:
                delta_t = -y[i]
                delta_beta = lambda_ * beta - y[i] * X[i]
            else:
                delta_t = 0
                delta_beta = lambda_ * beta

            beta = beta - eta_t * delta_beta
            alpha = alpha - eta_t * delta_t

    return beta, alpha

# %%
if False:
    """
    Part 2: Test your code with the provided training (200 samples) and test (600 samples) datasets, which are
    subsets of the MNIST data. Each dataset consists of 257 columns, with the first 256 columns representing
    the features, and the last column indicating the label (either 5 or 6).
    - coding5_train.csv
    - coding5_test.csv
    If you set yi = 1 for label 5 and yi = -1 for label 6, then after obtaining β and α, for a test point x_*,
    you'll predict: y_* = 5, if x_*^t β + α > 0; otherwise, 6.
    Report confusion tables on the training and test datasets.
    """
# %%

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Convert labels 5 -> 1, 6 -> -1
    y = np.where(y == 5, 1, -1)
    
    return X, y

def create_confusion_matrix(y_true, y_pred):
    # Convert predictions back to original labels (5 and 6)
    y_true = np.where(y_true == 1, 5, 6)
    y_pred = np.where(y_pred == 1, 5, 6)
    
    conf_matrix = pd.crosstab(
        y_true, y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    return conf_matrix

X_train, y_train = prepare_data('coding5_train.csv')
X_test, y_test = prepare_data('coding5_test.csv')

lambda_ = 0.1
T = 20
beta, alpha = pegasos(X_train, y_train, lambda_, T)

def predict(X, beta, alpha):
    return np.where(np.dot(X, beta) + alpha > 0, 1, -1)

y_train_pred = predict(X_train, beta, alpha)
y_test_pred = predict(X_test, beta, alpha)

print("Training Set Confusion Matrix:")
print(create_confusion_matrix(y_train, y_train_pred))
print("\nTest Set Confusion Matrix:")
print(create_confusion_matrix(y_test, y_test_pred))

train_accuracy = np.mean(y_train == y_train_pred) * 100
test_accuracy = np.mean(y_test == y_test_pred) * 100
print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# %%
