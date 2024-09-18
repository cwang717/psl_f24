# %% (PSL) Coding Assignment 1
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# %% Set the seed
np.random.seed(9601)

# %%
if False:
    """
    Part 1: Generate Data
    1. First generate the 20 centers from two-dimensional normal and randomly split them into two classes
    of 10. You can use any mean and covariance structure. You should not regenerate the centers.
    Use these 20 centers throughout this simulation study.
    2. Given the 20 centers, generate a training sample of size 200 (100 from each class) and a test sample of
    size 10,000 (5,000 from each class).
    3. Produce a scatter plot of the training data:
    • assign different colors to the two classes of data points;
    • overlay the 20 centers on this scatter plot, using a distinguishing
    """

# Generate the 20 centers from two-dimensional normal and split them into two classes of 10.
mean = np.random.randn(20, 2)
variance = 0.2

class_1_mean = mean[:10]
class_2_mean = mean[10:]

# Generate a training sample of size 200 (100 from each class) and a test sample of size 10,000 (5,000 from each class).
class_1_training_X = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=10) for mean in class_1_mean])
class_2_training_X = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=10) for mean in class_2_mean])

class_1_test_X = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=500) for mean in class_1_mean])
class_2_test_X = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=500) for mean in class_2_mean])

class_1_training_y = np.zeros(100)
class_2_training_y = np.ones(100)

class_1_test_y = np.zeros(5000)
class_2_test_y = np.ones(5000)

# Combine the training samples and test samples
training_sample_X = np.concatenate((class_1_training_X, class_2_training_X))
training_sample_y = np.concatenate((class_1_training_y, class_2_training_y))

test_sample_X = np.concatenate((class_1_test_X, class_2_test_X))
test_sample_y = np.concatenate((class_1_test_y, class_2_test_y))

# plot a scatter plot of the training sample, colored by class
plt.figure(figsize=(10, 8))
plt.scatter(training_sample_X[:100, 0], training_sample_X[:100, 1], c='blue', label='Class 1')
plt.scatter(training_sample_X[100:, 0], training_sample_X[100:, 1], c='red', label='Class 2')
plt.scatter(class_1_mean[:, 0], class_1_mean[:, 1], c='darkblue', marker='x', s=100, label='Class 1 Mean')
plt.scatter(class_2_mean[:, 0], class_2_mean[:, 1], c='darkred', marker='x', s=100, label='Class 2 Mean')
plt.title('Scatter Plot of Training Sample')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%
if False:
    """
    Part 2: kNN
    """

# %%
if False:
    """
    Part 2.1: Implement kNN from scratch; use Euclidean Distance. Your implementation should meet the following
    requirements:
    • Input: Your kNN function should accept three input parameters: training data (with labels), test
    data (without labels), and k. No need to write your kNN function to handle any general input; it
    suffices to write a function that is able to handle the data for this specific simulation study: binary
    classification; features are two-dimensional numerical vectors.
    • Output: Your function should return a vector of predictions for the test data.
    • Vectorization: Efficiently compute distances between all test points and training points simultaneously.
    Make predictions for all test points in a single operation.
    • No Loops: Do not use explicit loops like for or while inside your kNN function to compute
    distances or make predictions. Instead, harness the power of vectorized operations for efficient
    
    Part 2.2: Explain how you handle distance ties and voting ties
    See code comments
    """
# %%
def knn(X_train, y_train, X_test, k):
    # Compute pairwise squared Euclidean distances
    train_squared = np.sum(X_train**2, axis=1)[:, np.newaxis]
    test_squared = np.sum(X_test**2, axis=1)[np.newaxis, :]
    cross_term = np.dot(X_train, X_test.T)
    distances = np.sqrt(train_squared + test_squared - 2 * cross_term)
    
    # Add small random noise to break distance ties
    epsilon = 1e-8
    distances += np.random.uniform(0, epsilon, size=distances.shape)
    
    # Get k nearest neighbors
    k_nearest_indices = np.argpartition(distances, k, axis=0)[:k]
    k_nearest_ys = y_train[k_nearest_indices]
    
    # Handle voting ties by using a random tiebreaker
    vote_counts = np.sum(k_nearest_ys, axis=0)
    predictions = np.where(vote_counts == k/2,
                           np.random.randint(2, size=vote_counts.shape),
                           vote_counts > k/2)
    
    return predictions.astype(int)

# %% 
if False:
    """
    Part 2.3: Test your code with the training/test data you just generated when K = 1, 3, 5; and compare your
results with knn in R or sklearn.neighbors in Python.
    • Report your results (on the test data) as a 2-by-2 table (confusion matrix) for each K value.
    • Report the results from knn or sklearn.neighbors as a 2-by-2 table (confusion matrix) for each
    K value.
    """

def print_confusion_matrix(y_true, y_pred, k):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for K = {k}:")
    print(cm)
    print(f"Accuracy: {np.mean(y_true == y_pred):.4f}\n")

k_values = [1, 3, 5]

print("My kNN Implementation Results:")
for k in k_values:
    predictions = knn(training_sample_X, training_sample_y, test_sample_X, k)
    print_confusion_matrix(test_sample_y, predictions, k)

# Test sklearn's KNeighborsClassifier
print("sklearn KNeighborsClassifier Results:")
for k in k_values:
    sklearn_knn = KNeighborsClassifier(n_neighbors=k)
    sklearn_knn.fit(training_sample_X, training_sample_y)
    sklearn_predictions = sklearn_knn.predict(test_sample_X)
    print_confusion_matrix(test_sample_y, sklearn_predictions, k)

# %%
if False:
    """
    Part 3: cvKNN

    Part 3.1: Implement KNN classification with K chosen by 10-fold cross-validation from scratch.
    • Set the candidate K values from 1 to 180. (The maximum candidate K value is 180. Why?)
    • From now on, you are allowed to use the built-in kNN function from R or Python instead of your
    own implementation from Part 2.
    • It is possible that multiple K values give the (same) smallest CV error; when this happens, pick
    the largest K value among them, since the larger the K value, the simpler the model.
    """
# %%
def knn_cv(X, y, k_range, n_folds=10):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = np.zeros((len(k_range), n_folds))

    for i, k in enumerate(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        for j, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            cv_scores[i, j] = accuracy_score(y_val, y_pred)

    mean_cv_scores = np.mean(cv_scores, axis=1)
    best_k_indices = np.where(mean_cv_scores == np.max(mean_cv_scores))[0]
    best_k = k_range[best_k_indices[-1]]

    return int(best_k), mean_cv_scores

# Set candidate K values
k_range = range(1, 181)
best_k, cv_scores = knn_cv(training_sample_X, training_sample_y, k_range)

# Train final model with best K on entire training set
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(training_sample_X, training_sample_y)

test_predictions = final_model.predict(test_sample_X)

# Calculate and print accuracy
test_accuracy = accuracy_score(test_sample_y, test_predictions)
test_error = 1 - test_accuracy

print("10-Fold Cross-Validation Results:")
print(f"Best K value: {best_k}")
print(f"Test error: {test_error:.4f}")

cm = confusion_matrix(test_sample_y, test_predictions)
print("Confusion Matrix:")
print(cm)

# %% 
if False:
    """
    Part 4: Bayes Rule
    
    Part 4.1: Implement the Bayes rule. Your implementation should meet the following requirements:
    • Do not use explicit loops over the test sample size (10,000 or 5,000).
    • You are allowed to use loops over the number of centers (10 or 20), although you can avoid all
    loops.  
    """
# %%
def bayes_rule(X, class_means, class_priors, variance):
    num_classes = len(class_priors)
    num_centers = class_means.shape[0] // num_classes
    
    log_likelihoods = np.zeros((num_classes, X.shape[0]))
    
    for c in range(num_classes):
        class_mean = class_means[c*num_centers:(c+1)*num_centers]
        distances = np.sum((X[:, np.newaxis, :] - class_mean[np.newaxis, :, :])**2, axis=2)
        point_likelihoods = np.exp(-distances / (2 * variance)) / (2 * np.pi * variance)
        log_likelihoods[c] = np.log(np.sum(point_likelihoods, axis=1) / num_centers)
    
    log_posteriors = log_likelihoods + np.log(class_priors)[:, np.newaxis]
    
    # Compare log posteriors: predict 1 if log P(Y=1|X) > log P(Y=0|X)
    predictions = (log_posteriors[1] > log_posteriors[0]).astype(int)
    
    return predictions

# %%
if False:
    """
    Part 4.2: Test your code with the test data you just generated. (Note that you do not need training data for the
Bayes rule.) Report your results (on the test data) as a 2-by-2 table.
    """
class_means = np.vstack((class_1_mean, class_2_mean))
class_priors = np.array([0.5, 0.5])

bayes_predictions = bayes_rule(test_sample_X, class_means, class_priors, variance)

# Calculate accuracy and error
bayes_accuracy = np.mean(bayes_predictions == test_sample_y)
bayes_error = 1 - bayes_accuracy
print(f"Bayes Rule Error: {bayes_error:.4f}")

bayes_cm = confusion_matrix(test_sample_y, bayes_predictions)
print("Bayes Rule Confusion Matrix:")
print(bayes_cm)

# %%
if False:
    """
    Part 5: Simulation Study
    Given the 20 centers generated in Part 1, repeatedly generate 50 training/test datasets (training size = 200
    and test size = 10,000). For each pair of training/test datasets, calculate the test errors (the averaged 0/1
    loss on the test data set) for each of the following three procedures:
    1. kNN with K = 7 (you can use the built-in kNN function from R or Python);
    2. kNN with K chosen by 10-fold CV (your implementation from Part 3); and
    3. the Bayes rule (your implementation from Part 4).Part 4.3: Compare the test error rates of kNN (K = 7), kNN with K chosen by 10-fold cross-validation, and Bayes rule.
    """

def generate_dataset(class_means, variance, train_size, test_size):
    class_1_train = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=train_size//20) for mean in class_means[:10]])
    class_2_train = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=train_size//20) for mean in class_means[10:]])
    class_1_test = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=test_size//20) for mean in class_means[:10]])
    class_2_test = np.vstack([np.random.multivariate_normal(mean, np.eye(2) * variance, size=test_size//20) for mean in class_means[10:]])
    
    X_train = np.vstack((class_1_train, class_2_train))
    y_train = np.hstack((np.zeros(train_size//2), np.ones(train_size//2)))
    X_test = np.vstack((class_1_test, class_2_test))
    y_test = np.hstack((np.zeros(test_size//2), np.ones(test_size//2)))
    
    return X_train, y_train, X_test, y_test

# Simulation parameters
n_simulations = 50
train_size = 200
test_size = 10000
k_range = range(1, 181)

# Initialize arrays to store results
errors_knn7 = np.zeros(n_simulations)
errors_knn_cv = np.zeros(n_simulations)
errors_bayes = np.zeros(n_simulations)
selected_k_values = np.zeros(n_simulations, dtype=int)

for i in tqdm(range(n_simulations)):
    # Generate dataset
    X_train, y_train, X_test, y_test = generate_dataset(class_means, variance, train_size, test_size)
    
    # 1. kNN with K = 7
    knn7 = KNeighborsClassifier(n_neighbors=7)
    knn7.fit(X_train, y_train)
    y_pred_knn7 = knn7.predict(X_test)
    errors_knn7[i] = 1 - accuracy_score(y_test, y_pred_knn7)
    
    # 2. kNN with K chosen by 10-fold CV
    best_k, _ = knn_cv(X_train, y_train, k_range)
    selected_k_values[i] = best_k
    knn_cv_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_cv_model.fit(X_train, y_train)
    y_pred_knn_cv = knn_cv_model.predict(X_test)
    errors_knn_cv[i] = 1 - accuracy_score(y_test, y_pred_knn_cv)
    
    # 3. Bayes rule
    class_means = np.vstack((class_1_mean, class_2_mean))
    class_priors = np.array([0.5, 0.5])
    y_pred_bayes = bayes_rule(X_test, class_means, class_priors, variance)
    errors_bayes[i] = 1 - accuracy_score(y_test, y_pred_bayes)

k_min = np.min(selected_k_values)
k_max = np.max(selected_k_values)
k_median = np.median(selected_k_values)
k_25 = np.percentile(selected_k_values, 25)
k_75 = np.percentile(selected_k_values, 75)

print("\nFive-number summary for selected K values:")
print(f"Minimum: {k_min}")
print(f"25% quantile: {k_25}")
print(f"Median: {k_median}")
print(f"75% quantile: {k_75}")
print(f"Maximum: {k_max}")

# Print average test errors
print("\nAverage Test Errors:")
print(f"kNN (K=7): {np.mean(errors_knn7):.4f} ± {np.std(errors_knn7):.4f}")
print(f"kNN (CV): {np.mean(errors_knn_cv):.4f} ± {np.std(errors_knn_cv):.4f}")
print(f"Bayes rule: {np.mean(errors_bayes):.4f} ± {np.std(errors_bayes):.4f}")

# Plot test errors
plt.figure(figsize=(10, 6))
plt.boxplot([errors_knn7, errors_knn_cv, errors_bayes], labels=['kNN (K=7)', 'kNN (CV)', 'Bayes rule'])
plt.title('Test Errors for Different Classification Methods')
plt.ylabel('Test Error')
plt.show()

# %%