# %% (PSL) Quiz
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


np.random.seed(9601)

# %%

# Load the data with whitespace separator
auto_data = pd.read_csv('Q2Auto.data', delim_whitespace=True)

# Extract X (displacement) and y (mpg)
X = auto_data['displacement'].values.reshape(-1, 1)
y = auto_data['mpg'].values

# Fit linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.title('MPG vs Displacement Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Print regression results
print(f'Intercept: {model.intercept_:.2f}')
print(f'Slope: {model.coef_[0]:.4f}')
print(f'R-squared: {model.score(X, y):.4f}')

# %%
# Q2.3
# Prepare X (predictors) and y (response)
X = auto_data[['cylinders', 'displacement', 'horsepower', 'weight', 
               'acceleration', 'year', 'origin']]
y = auto_data['mpg']

# Fit multiple linear regression
model = LinearRegression()
model.fit(X, y)

# Print coefficients
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Print R-squared
print(f"\nR-squared: {model.score(X, y)}")
# %%
# Q2.4
# Linear model predicts starting salary after graduation 

# Given values
gpa = 4.0
iq = 100
gender = 1  # Female

# Calculate interaction terms
gpa_iq = gpa * iq
gpa_gender = gpa * gender

# Coefficients
b0 = 50
b1 = 20
b2 = 0.07
b3 = 35
b4 = 0.01
b5 = -10

# Calculate predicted salary
salary = b0 + b1*gpa + b2*iq + b3*gender + b4*gpa_iq + b5*gpa_gender
print(f"Predicted salary: ${salary}k")
# %%
# Q3.11
# Linear model of the "prostate" data. Load the data and split the data into training and test by the variable "train". The training set should contain 67 samples and the test contains 30 samples.

# Load the data
prostate_data = pd.read_csv('prostate.data', delim_whitespace=True)

# Split the data into training and test
train_data = prostate_data[prostate_data['train'] == 'T']
test_data = prostate_data[prostate_data['train'] == 'F']

# Prepare features (X) and target (y) for training
X_train = train_data.drop(['lpsa', 'train'], axis=1)
y_train = train_data['lpsa']

# Prepare features (X) and target (y) for testing
X_test = test_data.drop(['lpsa', 'train'], axis=1)
y_test = test_data['lpsa']

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate test error (sum of squared differences)
test_error = np.sum((y_test - y_pred) ** 2)

print(f"Test error (sum of squared differences): {test_error:.4f}")

# %%
# Now use AIC to find the best model

def calculate_aic(y_true, y_pred, n_params):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    aic = n * np.log(mse) + 2 * n_params
    return aic

features = X_train.columns
best_aic = float('inf')
best_features = None

# Try all possible combinations of features
for i in range(1, len(features) + 1):
    for combo in combinations(features, i):
        # Fit model with current feature combination
        X_train_subset = X_train[list(combo)]
        model = LinearRegression()
        model.fit(X_train_subset, y_train)
        
        # Calculate AIC
        y_pred = model.predict(X_train_subset)
        current_aic = calculate_aic(y_train, y_pred, i + 1)  # +1 for intercept
        
        # Update best model if current AIC is lower
        if current_aic < best_aic:
            best_aic = current_aic
            best_features = combo

print(f"Best features selected by AIC: {best_features}")
print(f"Number of features in optimal model: {len(best_features)}")

# For AIC-selected model
X_train_aic = X_train[list(best_features)]
X_test_aic = X_test[list(best_features)]
model_aic = LinearRegression()
model_aic.fit(X_train_aic, y_train)
y_pred_aic = model_aic.predict(X_test_aic)
test_error_aic = np.sum((y_test - y_pred_aic) ** 2)
print(f"Test error (sum of squared differences) for AIC-selected model: {test_error_aic:.4f}")
# %%
# Now use BIC to find the best model

def calculate_bic(y_true, y_pred, n_params):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    bic = n * np.log(mse) + np.log(n) * n_params
    return bic

features = X_train.columns
best_bic = float('inf')
best_features = None

# Try all possible combinations of features
for i in range(1, len(features) + 1):
    for combo in combinations(features, i):
        # Fit model with current feature combination
        X_train_subset = X_train[list(combo)]
        model = LinearRegression()
        model.fit(X_train_subset, y_train)
        
        # Calculate BIC
        y_pred = model.predict(X_train_subset)
        current_bic = calculate_bic(y_train, y_pred, i + 1)  # +1 for intercept
        
        # Update best model if current BIC is lower
        if current_bic < best_bic:
            best_bic = current_bic
            best_features = combo

print(f"Best features selected by BIC: {best_features}")
print(f"Number of features in optimal model: {len(best_features)}")

# For BIC-selected model
X_train_bic = X_train[list(best_features)]
X_test_bic = X_test[list(best_features)]
model_bic = LinearRegression()
model_bic.fit(X_train_bic, y_train)
y_pred_bic = model_bic.predict(X_test_bic)
test_error_bic = np.sum((y_test - y_pred_bic) ** 2)
print(f"Test error (sum of squared differences) for BIC-selected model: {test_error_bic:.4f}")
# %%
# Q5.8
# cubic polynomial regression

# Load the data
nox_data = pd.read_csv('noxData.csv')

# Extract X (dis) and y (nox)
X = nox_data['dis'].values.reshape(-1, 1)
y = nox_data['nox'].values

# Create polynomial features (cubic: x, x², x³)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Fit polynomial regression
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Calculate residual sum of squares
rss = np.sum((y - y_pred) ** 2)
print(f"Residual Sum of Squares: {rss:.6f}")

# Optional: Create plot to visualize the fit
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
X_sort = np.sort(X, axis=0)
X_sort_poly = poly.transform(X_sort)
y_sort = model.predict(X_sort_poly)
plt.plot(X_sort, y_sort, color='red', label='Cubic regression')
plt.xlabel('dis')
plt.ylabel('nox')
plt.title('Cubic Polynomial Regression: NOX vs Distance')
plt.legend()
plt.grid(True)
plt.show()

# Create the feature value for dis=6
X_new = np.array([[6]])
X_new_poly = poly.transform(X_new)  # Transform to polynomial features

# Make prediction
prediction = model.predict(X_new_poly)
print(f"Predicted NOX when dis=6: {prediction[0]:.6f}")

# %%
# Get the design matrix (X_poly)
X = nox_data['dis'].values.reshape(-1, 1)
X_poly = poly.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Calculate standard errors
n = len(y)
y_pred = model.predict(X_poly)
mse = np.sum((y - y_pred) ** 2) / (n - 4)  # 4 parameters (including intercept)
var_coef = mse * np.linalg.inv(X_poly.T @ X_poly).diagonal()
se = np.sqrt(var_coef)

# Calculate t-statistics
t_stats = model.coef_ / se

# Calculate p-values
p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=n-4))

# The cubic term is the last coefficient
cubic_p_value = p_values[-1]
print(f"P-value for cubic term: {cubic_p_value:.6f}")
# %%
# fit a fourth-degree polynomial regression model.

# Q5.8 (continued)
# fourth-degree polynomial regression

# Load the data (if not already loaded)
X = nox_data['dis'].values.reshape(-1, 1)
y = nox_data['nox'].values

# Create polynomial features (fourth degree: x, x², x³, x⁴)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Fit polynomial regression
model = LinearRegression()
model.fit(X_poly, y)

# Make predictions
y_pred = model.predict(X_poly)

# Calculate residual sum of squares
rss = np.sum((y - y_pred) ** 2)
print(f"Residual Sum of Squares (4th degree): {rss:.6f}")

# Create plot to visualize the fit
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
X_sort = np.sort(X, axis=0)
X_sort_poly = poly.transform(X_sort)
y_sort = model.predict(X_sort_poly)
plt.plot(X_sort, y_sort, color='red', label='4th degree polynomial')
plt.xlabel('dis')
plt.ylabel('nox')
plt.title('Fourth-Degree Polynomial Regression: NOX vs Distance')
plt.legend()
plt.grid(True)
plt.show()

# Calculate standard errors and p-values
n = len(y)
mse = np.sum((y - y_pred) ** 2) / (n - 5)  # 5 parameters (including intercept)
var_coef = mse * np.linalg.inv(X_poly.T @ X_poly).diagonal()
se = np.sqrt(var_coef)

# Calculate t-statistics
t_stats = model.coef_ / se

# Calculate p-values
p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=n-5))

# The fourth-degree term is the last coefficient
fourth_degree_p_value = p_values[-1]
print(f"P-value for fourth-degree term: {fourth_degree_p_value:.6f}")

# Create the feature value for dis=6
X_new = np.array([[6]])
X_new_poly = poly.transform(X_new)  # Transform to polynomial features

# Make prediction
prediction = model.predict(X_new_poly)
print(f"Predicted NOX when dis=6: {prediction[0]:.6f}")

# %%
# LDA classifier
train = pd.read_csv("zip.train", sep=' ', header=None)
Y = train.iloc[:, 0]
X = train.iloc[:, 1:]

# Load test data
test = pd.read_csv("zip.test", sep=' ', header=None)
Y_test = test.iloc[:, 0]
X_test = test.iloc[:, 1:]

# Train LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X, Y)

# Get predictions
Y_pred = lda.predict(X_test)

# Get probability estimates
proba = lda.predict_proba(X_test)

# [A] Total number of digit "4" in test set
A = sum(Y_test == 4)

# [B] Number of correctly classified digit "4"
B = sum((Y_test == 4) & (Y_pred == 4))

# [C] Number of images classified as "4"
C = sum(Y_pred == 4)

# [D] Number of misclassified images among those classified as "4"
D = sum((Y_pred == 4) & (Y_test != 4))

# [E] Actual digit of 4th image
E = Y_test.iloc[3]

# Get probabilities for 4th image
fourth_image_probs = proba[3]
sorted_digits = np.argsort(-fourth_image_probs)  # Sort in descending order

# [F] Most probable digit for 4th image
F = sorted_digits[0]

# [G] Second most probable digit for 4th image
G = sorted_digits[1]

print(f"A: {A}")
print(f"B: {B}")
print(f"C: {C}")
print(f"D: {D}")
print(f"E: {E}")
print(f"F: {F}")
print(f"G: {G}")
# %%
# Q9.6
# Linear SVM for spam classification

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset
spam = pd.read_table("spam.txt", sep="\s+", header=None)

testID = list(range(0, 100)) + list(range(1900, 1960))
spam_test = spam.loc[testID, :]
spam_train = spam.drop(testID)
X = spam_train.drop(57, axis=1)
y = spam_train[57]
Xtest = spam_test.drop(57, axis=1)
ytest = spam_test[57]

sc = StandardScaler()
sc.fit(X)
sc.scale_ = np.std(X, axis=0, ddof=1).tolist()
newX = sc.transform(X)
newXtest = sc.transform(Xtest)

# Define the different cost values to evaluate
cost_values = [1, 10, 50]

for cost in cost_values:
    clf = SVC(kernel='linear', C=cost)
    clf.fit(newX, y)
    
    # Predictions on training data
    y_pred_train = clf.predict(newX)
    ctable_train = confusion_matrix(y, y_pred_train)
    train_errors = round(ctable_train[0, 1] + ctable_train[1, 0])
    
    # Predictions on test data
    y_pred_test = clf.predict(newXtest)
    ctable_test = confusion_matrix(ytest, y_pred_test)
    test_errors = round(ctable_test[0, 1] + ctable_test[1, 0])
    
    # Number of support vectors
    num_SVs = round(sum(clf.n_support_))
    
    print(f"\nFor cost={cost}:")
    print(f"Number of Support Vectors: {num_SVs}")
    print(f"Training Error: {train_errors}")
    print(f"Test Error: {test_errors}")
# %%
# Gaussian SVM for spam classification

# Use the same data preparation as before
# Train Gaussian SVM with different cost values
for cost in [1, 10, 50]:
    clf = SVC(kernel='rbf', C=cost)  # Using default gamma
    clf.fit(newX, y)
    
    # Predictions on training data
    y_pred_train = clf.predict(newX)
    ctable_train = confusion_matrix(y, y_pred_train)
    train_errors = round(ctable_train[0, 1] + ctable_train[1, 0])
    
    # Predictions on test data
    y_pred_test = clf.predict(newXtest)
    ctable_test = confusion_matrix(ytest, y_pred_test)
    test_errors = round(ctable_test[0, 1] + ctable_test[1, 0])
    
    # Number of support vectors
    num_SVs = round(sum(clf.n_support_))
    
    print(f"\nFor cost={cost}:")
    print(f"Number of Support Vectors: {num_SVs}")
    print(f"Training Error: {train_errors}")
    print(f"Test Error: {test_errors}")
# %%
# Q8


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf


# Load the dataset
url = "https://liangfgithub.github.io/Data/Caravan.csv"
data = pd.read_csv(url)

# Rename the last column to "Y"
data.rename(columns={data.columns[-1]: "Y"}, inplace=True)

# Convert "Y" column to binary values (1 for "Yes", 0 for "No")
data['Y'] = data['Y'].apply(lambda x: 1 if x == "Yes" else 0)

# Convert "Y" column to a factor
data['Y'] = pd.Categorical(data['Y'])


# Split data into train and test sets
train = data[1000:]
test = data[:1000]

X = train.drop(columns=['Y'])
y = train['Y']
X = sm.add_constant(X)

X_test = test.drop(columns=['Y'])
y_test = test['Y']
X_test = sm.add_constant(X_test)

# Logistic regression
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
#result.summary()

y_pred_prob = result.predict(X_test)
threshold = 0.25
y_pred_label = (y_pred_prob > threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_label)
auc = roc_auc_score(y_test, y_pred_prob)

print("Confusion Matrix:")
print(conf_matrix)
print("AUC:", auc)

# For "No" (0) class - false positives
a1 = conf_matrix[0][1]  # misclassified "No" samples

# For "Yes" (1) class - false negatives
b1 = conf_matrix[1][0]  # misclassified "Yes" samples

# Format AUC to 3 decimal places
c1 = round(auc, 3)

print("\nAnswers:")
print(f"a1 (misclassified 'No' samples): {a1}")
print(f"b1 (misclassified 'Yes' samples): {b1}")
print(f"c1 (AUC to 3 decimal places): {c1}")

# %%

def computeAIC(feature_set):
    model = sm.GLM(y, X[list(feature_set)], family=sm.families.Binomial())
    result = model.fit()
    AIC = result.aic
    return {"model":result, "AIC":AIC}

def computeBIC(feature_set):
    model = sm.GLM(y, X[list(feature_set)], family=sm.families.Binomial())
    result = model.fit()
    BIC = result.aic + len(feature_set)*(np.log(len(y))-2)
    return {"model":result, "BIC":BIC}

def AICforward(features):
    remaining_features = [p for p in X.columns if p not in features]

    results = []
    
    for p in remaining_features:
        results.append(computeAIC(features+[p]))
    
    models = pd.DataFrame(results)
    best_model = models.loc[models['AIC'].argmin()]
    
    return best_model

def BICforward(features):
    remaining_features = [p for p in X.columns if p not in features]

    results = []
    
    for p in remaining_features:
        results.append(computeBIC(features+[p]))
    
    models = pd.DataFrame(results)
    best_model = models.loc[models['BIC'].argmin()]
    
    return best_model

AIC_fwd = pd.DataFrame(columns=["AIC", "model"])
features = ['const']

best_AIC = 0

for i in range(1,len(X.columns)): 
    AIC = AICforward(features)
    if i==1:
        best_AIC = AIC['AIC']
    if best_AIC < AIC['AIC']:
        break
    best_AIC = AIC['AIC']
    AIC_fwd.loc[i] = AIC
    features = AIC_fwd.loc[i]["model"].model.exog_names




AIC_var_list = AIC_fwd.loc[17]["model"].model.exog_names
model = sm.GLM(y, X[list(AIC_var_list)], family=sm.families.Binomial())
result = model.fit()

y_pred_prob = result.predict(X_test[list(AIC_var_list)])
threshold = 0.25
y_pred_label = (y_pred_prob > threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_label)
auc = roc_auc_score(y_test, y_pred_prob)

print("Confusion Matrix:")
print(conf_matrix)
print("AUC:", auc)

# Calculate number of non-intercept predictors (d2)
d2 = len(AIC_var_list) - 1  # subtract 1 to exclude intercept ('const')

# Get misclassification counts from confusion matrix
# Confusion matrix format:
# [[TN  FP]
#  [FN  TP]]

# For "No" (0) class - false positives
a2 = conf_matrix[0][1]  # misclassified "No" samples

# For "Yes" (1) class - false negatives
b2 = conf_matrix[1][0]  # misclassified "Yes" samples

# Format AUC to 3 decimal places
c2 = round(auc, 3)

print("\nAnswers for AIC forward selection model:")
print(f"d2 (number of non-intercept predictors): {d2}")
print(f"a2 (misclassified 'No' samples): {a2}")
print(f"b2 (misclassified 'Yes' samples): {b2}")
print(f"c2 (AUC to 3 decimal places): {c2}")

# %%
# BIC forward selection
BIC_fwd = pd.DataFrame(columns=["BIC", "model"])
features = ['const']

best_BIC = 0

for i in range(1,len(X.columns)): 
    BIC = BICforward(features)
    if i==1:
        best_BIC = BIC['BIC']
    if best_BIC < BIC['BIC']:
        break
    best_BIC = BIC['BIC']
    BIC_fwd.loc[i] = BIC
    features = BIC_fwd.loc[i]["model"].model.exog_names

# Get the final model variables and fit
BIC_var_list = BIC_fwd.loc[len(BIC_fwd.index)]["model"].model.exog_names
model = sm.GLM(y, X[list(BIC_var_list)], family=sm.families.Binomial())
result = model.fit()

# Make predictions
y_pred_prob = result.predict(X_test[list(BIC_var_list)])
threshold = 0.25
y_pred_label = (y_pred_prob > threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_label)
auc = roc_auc_score(y_test, y_pred_prob)

# Calculate answers
d3 = len(BIC_var_list) - 1  # subtract 1 to exclude intercept ('const')
a3 = conf_matrix[0][1]  # misclassified "No" samples
b3 = conf_matrix[1][0]  # misclassified "Yes" samples
c3 = round(auc, 3)

print("\nAnswers for BIC forward selection model:")
print(f"d3 (number of non-intercept predictors): {d3}")
print(f"a3 (misclassified 'No' samples): {a3}")
print(f"b3 (misclassified 'Yes' samples): {b3}")
print(f"c3 (AUC to 3 decimal places): {c3}")

# %%
# Logistic Regression with L1 Penalty
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = train.drop(columns=['Y'])
y = train['Y']

X_test = test.drop(columns=['Y'])
y_test = test['Y']

sc = StandardScaler()
sc.fit(X)
sc.scale_ = np.std(X, axis=0, ddof=1).to_list()
newX = sc.transform(X)
newX_test = sc.transform(X_test)

newy = y.cat.codes


cost = 1/(newX.shape[0] * 0.004)
clf = LogisticRegression(penalty='l1', C=cost, solver='liblinear')
clf.fit(newX, newy)

np.count_nonzero(clf.coef_)



y_pred_prob = clf.predict_proba(newX_test)[:,1]
threshold = 0.25
y_pred_label = (y_pred_prob > threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_label)
auc = roc_auc_score(y_test, y_pred_prob)

print("Confusion Matrix:")
print(conf_matrix)
print("AUC:", auc)

# Calculate number of non-zero coefficients (excluding intercept)
d4 = np.count_nonzero(clf.coef_)

# Get misclassification counts from confusion matrix
# Confusion matrix format:
# [[TN  FP]
#  [FN  TP]]

# For "No" (0) class - false positives
a4 = conf_matrix[0][1]  # misclassified "No" samples

# For "Yes" (1) class - false negatives
b4 = conf_matrix[1][0]  # misclassified "Yes" samples

# Format AUC to 3 decimal places
c4 = round(auc, 3)

print("\nAnswers for L1 penalized model:")
print(f"d4 (number of non-intercept predictors): {d4}")
print(f"a4 (misclassified 'No' samples): {a4}")
print(f"b4 (misclassified 'Yes' samples): {b4}")
print(f"c4 (AUC to 3 decimal places): {c4}")

# %%
