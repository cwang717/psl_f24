# %% (PSL) Coding Assignment 2
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# %% Set the seed
np.random.seed(9601)

# %%
if False:
    """
    Part 1: Implement Lasso
    
    1.1 One-variable Lasso
    First, write a function one_var_lasso that takes the following inputs:
    - v: a vector of length n
    - z: a vector of length n
    - lambda > 0
    and solves the following one-variable Lasso problem (b is a scalar)
    min_b 1/2n ||v - b z||^2 + lambda |b|
    """

# %%
def one_var_lasso(r, x, lam):
    n = len(r)
    x_norm_squared = np.dot(x.T, x)[0, 0]
    r_dot_x = np.dot(r.T, x)[0, 0]
    
    if abs(r_dot_x) <= n * lam:
        return 0
    else:
        return np.sign(r_dot_x) * (abs(r_dot_x) - n * lam) / x_norm_squared

# %%
if False:
    """
    1.2 The CD Alogirthm

    Next, write your own function MyLasso to implement the Coordinate Descent (CD) algorithm by
    repeatedly calling one_var_lasso.
    In the CD algorithm, at each iteration, we solve a one-variable Lasso problem for βj while holding the other
    (p-1) coefficients at their current values.
    """

# %%
def MyLasso(X, y, lam_seq, maxit = 100):
    
    # Input
    # X: n-by-p design matrix without the intercept 
    # y: n-by-1 response vector 
    # lam.seq: sequence of lambda values (arranged from large to small)
    # maxit: number of updates for each lambda 
    
    # Output
    # B: a (p+1)-by-len(lam.seq) coefficient matrix 
    #    with the first row being the intercept sequence 

    n, p = X.shape
    nlam = len(lam_seq)
    B = np.zeros((p+1, nlam))
    
    ##############################
    # YOUR CODE: 
    # (1) newX = Standardizad X; 
    # (2) Record the centers and scales used in (1) 
    ##############################
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    newX = (X - X_mean) / X_std
    y_mean = np.mean(y)
    y_centered = y - y_mean

    # Initilize coef vector b and residual vector r
    b = np.zeros(p)
    r = y_centered

    # Triple nested loop
    for m in range(nlam):
        for step in range(maxit):
            for j in range(p):
                X_j = newX[:, j].reshape(-1,1)
                r = r + X_j * b[j]
                b[j] = one_var_lasso(r, X_j, lam_seq[m])
                r = r - X_j * b[j]
        B[1:, m] = b 
    
    ##############################
    # YOUR CODE:
    # Scale back the coefficients;
    # Update the intercepts stored in B[0, ]
    ##############################
    B[1:, :] = B[1:, :] / X_std.reshape(-1, 1)
    B[0, :] = y_mean - np.dot(X_mean, B[1:, :])
    
    return(B)

# %%
if False:
    """
    1.3 Test your Function
    """

# %%
# Load Data
myData = pd.read_csv("Coding2_Data0.csv")
var_names = myData.columns
y = myData[['Y']].to_numpy()
X = myData.drop(['Y'], axis = 1).to_numpy()

print(X.shape, len(y))

# create lambda sequence and test MyLasso
log_lam_seq = np.linspace(-1, -8, num = 80)
lam_seq = np.exp(log_lam_seq)
myout = MyLasso(X, y, lam_seq, maxit = 100)

# plot the lasso paths
p, _ = myout.shape
plt.figure(figsize = (12,8))

for i in range(p-1):
    plt.plot(log_lam_seq, myout[i+1, :], label = var_names[i])

plt.xlabel('Log Lambda')
plt.ylabel('Coefficients')
plt.title('Lasso Paths - Numpy implementation')
plt.legend()
plt.axis('tight')
plt.show()

# Check the Accuracy
lasso_coef = pd.read_csv("Coding2_lasso_coefs.csv").to_numpy()
print(lasso_coef.shape)

print("The maximum difference between myout and lasso_coef is:")
print(abs(myout - lasso_coef).max())

# %%
if False:
    """
    Part 2: Simulation Study
    Consider the following six procedures:
        • Full: Fit a linear regression model using all features
        • Ridge.min : Ridge regression using lambda.min
        • Lasso.min and Lasso.1se: Lasso using lambda.min or lambda.1se
        • L.Refit: Refit the model selected by Lasso using lambda.1se
        • PCR: principle components regression with the number of components chosen by 10-fold
        cross validation
    """

# %%
if False:
    """
    2.1 Case 1
    Download the data set Coding2_Data1.csv. The first 14 columns are the same as the data set we used in
    Part I with Y being the response variable (moved to the 1st column). The additional 78 more predictors are
    the quadratic and pairwise product terms of the original 13 predictors.
    - [a] Conduct the following simulation exercise 50 times:
        - In each iteration, randomly split the data into two parts, 75% for training and 25% for testing.
        - For each of the six procedures, train a model using the training subset and generate predictions
        for the test subset. Record the MSPE (Mean Squared Prediction Error) based on these test data
        predictions.
    - [b] Report the average MSPEs for the six procedures.
    - [c] Based on the outcomes of your simulation study, please address the following questions:
        - Which procedure or procedures yield the best performance in terms of MSPE?
        - Conversely, which procedure or procedures show the poorest performance?
        - In the context of Lasso regression, which procedure, Lasso.min or Lasso.1se, yields a better
        MSPE?
        - Is refitting advantageous in this case? In other words, does L.Refit outperform Lasso.1se?
        - Is variable selection or shrinkage warranted for this particular dataset? To clarify, do you find
        the performance of the Full model to be comparable to, or divergent from, the best-performing
        procedure among the other five?
    """
# %%
# PCR implementation
class PCR(object):

    def __init__(self, num_folds=10):
        self.folds = num_folds

    def fit(self, X, Y):
        n, p = X.shape
        indices = np.arange(n)
        np.random.shuffle(indices)
        index_sets = np.array_split(indices, self.folds)
        ncomp = min(p, n - 1 - max([len(i) for i in index_sets]))
        cv_err = np.zeros(ncomp)

        for ifold in range(self.folds):
            train_inds = np.concatenate(index_sets[:ifold] + index_sets[ifold+1:])
            test_inds = index_sets[ifold]

            X_train = X[train_inds, :]
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA())])
            pipeline.fit(X_train)
            X_train = pipeline.transform(X_train)
            coefs = Y[train_inds].T @ X_train / np.sum(X_train**2, axis=0)
            b0 = np.mean(Y[train_inds])

            X_test = pipeline.transform(X[test_inds, :])

            for k in np.arange(ncomp):
                preds = X_test[:, :k] @ coefs.T[:k] + b0
                cv_err[k] +=  np.sum((Y[test_inds]-preds)**2)

        min_ind = np.argmin(cv_err)
        self.ncomp = min_ind+1
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=self.ncomp))])
        self.transform = pipeline.fit(X)
        self.model = lm().fit(self.transform.transform(X), Y)

    def predict(self, X):
        X_ = self.transform.transform(X)
        return self.model.predict(X_)

# %%
# Load Data
myData = pd.read_csv("Coding2_Data1.csv")
Y = myData['Y'].to_numpy()
X = myData.drop(['Y'], axis=1).to_numpy()
print(X.shape, len(Y))

# Function to run one simulation
def run_simulation(skip_full = False):
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    if not skip_full:
        # Full Model
        full = lm().fit(X_train_scaled, Y_train)
        results['Full'] = mean_squared_error(Y_test, full.predict(X_test_scaled))
    
    # Ridge Regression
    ridge_alphas = np.logspace(-10, 1, 100)
    ridgecv = RidgeCV(alphas=ridge_alphas, cv=10, scoring='neg_mean_squared_error')
    ridgecv.fit(X_train_scaled, Y_train)
    ridge_model = Ridge(alpha=ridgecv.alpha_)
    ridge_model.fit(X_train_scaled, Y_train)
    results['Ridge.min'] = mean_squared_error(Y_test, ridge_model.predict(X_test_scaled))
    
    # Lasso
    lasso_alphas = np.logspace(-10, 1, 100)
    lassocv = LassoCV(alphas=lasso_alphas, cv=10)
    lassocv.fit(X_train_scaled, Y_train)

    mean_mse = np.mean(lassocv.mse_path_, axis=1)
    std_mse = np.std(lassocv.mse_path_, axis=1) / np.sqrt(10)

    cv_alphas = lassocv.alphas_
    min_idx = np.argmin(mean_mse)

    alpha_min = cv_alphas[min_idx]

    threshold = mean_mse[min_idx] + std_mse[min_idx]
    alpha_1se = max(cv_alphas[np.where(mean_mse <= threshold)])
    
    # Lasso.min
    lasso_model_min = Lasso(alpha=alpha_min)
    lasso_model_min.fit(X_train_scaled, Y_train)
    results['Lasso.min'] = mean_squared_error(Y_test, lasso_model_min.predict(X_test_scaled))
    
    # Lasso.1se
    lasso_model_1se = Lasso(alpha=alpha_1se)
    lasso_model_1se.fit(X_train_scaled, Y_train)
    results['Lasso.1se'] = mean_squared_error(Y_test, lasso_model_1se.predict(X_test_scaled))
    
    # L.Refit
    nonzero_indices = np.where(lasso_model_1se.coef_ != 0)[0]
    lm_refit = lm()
    lm_refit.fit(X_train_scaled[:, nonzero_indices], Y_train)
    results['L.Refit'] = mean_squared_error(Y_test, lm_refit.predict(X_test_scaled[:, nonzero_indices]))
    
    # PCR
    pcr = PCR()
    pcr.fit(X_train, Y_train)  # PCR already includes scaling in its pipeline
    results['PCR'] = mean_squared_error(Y_test, pcr.predict(X_test))
    
    return results

# Run simulations
n_simulations = 50
all_results = []

for _ in tqdm(range(n_simulations)):
    all_results.append(run_simulation())

# Calculate average MSPEs
average_mspe = {key: np.mean([result[key] for result in all_results]) for key in all_results[0].keys()}

# Print results
print("\nAverage MSPEs for the six procedures:")
for procedure, mspe in average_mspe.items():
    print(f"{procedure}: {mspe:.4f}")

# %%
# Analyze the results
# Find the best and worst performing procedures
best_procedure = min(average_mspe, key=average_mspe.get)
worst_procedure = max(average_mspe, key=average_mspe.get)

# Compare Lasso.min and Lasso.1se
lasso_comparison = "Lasso.min" if average_mspe["Lasso.min"] < average_mspe["Lasso.1se"] else "Lasso.1se"

# Compare L.Refit and Lasso.1se
refit_advantage = "Yes" if average_mspe["L.Refit"] < average_mspe["Lasso.1se"] else "No"

# Analyze if variable selection or shrinkage is warranted
best_among_five = min({k: v for k, v in average_mspe.items() if k != "Full"}, key=average_mspe.get)
selection_shrinkage_warranted = "Yes" if average_mspe[best_among_five] < average_mspe.get("Full", float('inf')) else "No"

print("\nAnswers to the questions:")
print(f"1. Best performing procedure(s): {best_procedure}")
print(f"2. Poorest performing procedure(s): {worst_procedure}")
print(f"3. Better MSPE in Lasso regression: {lasso_comparison}")
print(f"4. Is refitting advantageous? {refit_advantage}")
print(f"5. Is variable selection or shrinkage warranted? {selection_shrinkage_warranted}")


# %%
if False:
    """
    2.2 Case 2

    Download the data set Coding2_Data2.csv. The first 92 columns are identical to those in
    Coding2_Data1.csv, with the addition of 500 columns of artificially generated noise features.
        • Repeat [a] and [b] above for the five procedures excluding the Full procedure. 
        Graphically summarize your findings on MSPE using a strip chart, and consider overlaying a boxplot for additional insights.
        • [c] Address the following questions:
            - Which procedure or procedures yield the best performance in terms of MSPE?
            - Conversely, which procedure or procedures show the poorest performance?
            - In the context of Lasso regression, which procedure, Lasso.min or Lasso.1se, yields a better
            MSPE?
            - Is refitting advantageous in this case? In other words, does L.Refit outperform Lasso.1se?
            - Is variable selection or shrinkage warranted for this particular dataset? To clarify, do you find
            the performance of the Full model to be comparable to, or divergent from, the best-performing
            procedure among the other five?
    """

# %%
# Load Data for Case 2
myData = pd.read_csv("Coding2_Data2.csv")
Y = myData['Y'].to_numpy()
X = myData.drop(['Y'], axis=1).to_numpy()
print(X.shape, len(Y))

# Run simulations
n_simulations = 50
all_results = []

for _ in tqdm(range(n_simulations)):
    all_results.append(run_simulation(skip_full=True))

# Calculate average MSPEs
average_mspe = {key: np.mean([result[key] for result in all_results]) for key in all_results[0].keys()}

# Print results
print("\nAverage MSPEs for the five procedures:")
for procedure, mspe in average_mspe.items():
    print(f"{procedure}: {mspe:.4f}")

# Create a strip chart with overlaid boxplot
plt.figure(figsize=(12, 6))

data = [
    [result[procedure] for result in all_results]
    for procedure in all_results[0].keys()
]

positions = range(len(all_results[0].keys()))
labels = list(all_results[0].keys())

# Strip chart
for i, d in enumerate(data):
    plt.scatter([i] * len(d), d, alpha=0.4, color='blue')

# Boxplot
bp = plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black')

for patch in bp['boxes']:
    patch.set(facecolor='white', edgecolor='black', alpha=0.7)

plt.xticks(positions, labels, rotation=45)
plt.ylabel('MSPE')
plt.title('MSPE Distribution for Different Procedures (Case 2)')
plt.tight_layout()
plt.show()

# %%
# Analyze the results and answer the questions
best_procedure = min(average_mspe, key=average_mspe.get)
worst_procedure = max(average_mspe, key=average_mspe.get)

print("\nAnalysis of Case 2 results:")
print(f"1. Best performing procedure: {best_procedure} with MSPE of {average_mspe[best_procedure]:.4f}")
print(f"2. Worst performing procedure: {worst_procedure} with MSPE of {average_mspe[worst_procedure]:.4f}")

print("\n3. Comparison with Case 1:")
print("Lasso.min performs well in Case 2 but is not that good in Case 1; Ridge.min is the best in Case 1 but the worst in Case 2; PCR is good in Case 1 but not in Case 2.")
print("Possible explanations for performance changes: the 500 additional noise features in Case 2 may have made it harder for some procedures to identify relevant predictors.")
print("Therefore, procedures that tend to use more features (like Ridge.min or PCR) might be more prone to overfitting in Case 2.")
print("Lasso-based methods might perform relatively better in Case 2 due to their ability to perform feature selection.")

print("\n4. Comparison of best MSPEs between Case 1 and Case 2:")
print("The best MSPE in Case 2 is higher (worse) than in Case 1, despite having all the features from Case 1 plus additional ones.")
print("This can be explained by:")
print("   - Increased noise: The 500 additional noise features make it harder to identify the true underlying relationship.")
print("   - Curse of dimensionality: With many more features than observations, the risk of overfitting increases significantly.")
print("   - Model complexity: More complex models fitted to Case 2 data may not generalize well to unseen data.")
# %%
