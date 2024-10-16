# %% (PSL) Coding Assignment 3
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from skmisc.loess import loess

from scipy.interpolate import splev
from sklearn.cluster import KMeans

# %% Set the seed
np.random.seed(9601)

# %%
if False:
    """
    Part I: Optimal span for LOESS
    Objective
    Write your own function to employ LOO-CV and GCV in selecting the optimal span for LOESS. Definitions
    of LOO-CV and GCV can be found on page 33 of [lec_W5_NonlinearRegression.pdf]
    Below, we'll use CV to refer to LOO-CV.
    """

# %%
if False:
    """
    1.1 Retrieve the Diagonal of the Smoother Matrix
    Write a function to retrieve the Diagonal of the Smoother Matrix. We're only interested in the
    diagonal entries (which will be used in computing LOO-CV and GCV), so this function should return
    an n-by-1 vector.
    • Inputs: x (an n-by-1 feature vector) and span (a numerical value).
    • Output: n-by-1 vector representing the diagonal of the smoother matrix S.
    • If you're using Python, you can either use the aforementioned technique or directly retrieve the
    diagonal entries from the output of skmisc.loess [Link]. Keep in mind that the input to this
    function should only be x and span, not y. When calling skmisc.loess to retrieve the diagonal
    entries, you can use a fake y vector, as the diagonal entries remain the same and do not depend on
    the response vector y

    1.2 Write a function to find the Optimal Span(s) based on CV and GCV.
    • Iterate over the specified span values.
    • For each span, calculate the CV and GCV values.
    • Return the CV and GCV values corresponding to each span.
    • Determine the best span(s) based on the CV and GCV results.
    """

# %%
def loocv_for_loess_spans(x, y, span_values):
    loocv_values = []
    for span in span_values:
        loess_model = loess(x, y, span=span)
        loess_model.fit()
        loess_output = loess_model.outputs
        diagonals = loess_output.diagonal
        fitted_residuals = loess_output.fitted_residuals
        loocv = np.average([(residual/(1-diagonal))**2 for residual, diagonal in zip(fitted_residuals, diagonals)])
        loocv_values.append(loocv)
    return loocv_values

def best_span_using_loocv(x, y, span_values):
    loocv_values = loocv_for_loess_spans(x, y, span_values)
    print("LOO-CV values for each span:")
    for span, value in zip(span_values, loocv_values):
        print(f"Span: {span:.2f}, LOO-CV: {value:.4f}")
    best_span = span_values[np.argmin(loocv_values)]
    return best_span

# %%
def gcv_for_loess_spans(x, y, span_values):
    gcv_values = []
    for span in span_values:
        loess_model = loess(x, y, span=span)
        loess_model.fit()
        loess_output = loess_model.outputs
        diagonals = loess_output.diagonal
        average_diagonal = np.average(diagonals)
        fitted_residuals = loess_output.fitted_residuals
        gcv = np.average([(residual/(1-average_diagonal))**2 for residual in fitted_residuals])
        gcv_values.append(gcv)
    return gcv_values

def best_span_using_gcv(x, y, span_values):
    gcv_values = gcv_for_loess_spans(x, y, span_values)
    print("GCV values for each span:")
    for span, value in zip(span_values, gcv_values):
        print(f"Span: {span:.2f}, GCV: {value:.4f}")
    best_span = span_values[np.argmin(gcv_values)]
    return best_span

# %%
if False:
    """
    1.3 Test your code using the provided dataset [Coding3_Data.csv].
    • Report your CV and GCV for the following 15 span values: 0.20, 0.25, . . . , 0.90.
    • Determine the best span value based on the CV and GCV results. For this dataset, both methods
    recommend the same span.
    • Fit a LOESS model over the entire dataset using the selected optimal span.
    • Display the original data points and overlay them with the true curve and the fitted curve. Include
    a legend to distinguish between the two curves.
    • The true curve is
    f(x) = sin(12(x + 0.2))/(x + 0.2), x ∈ [0, 1]
    """

# %%
data = pd.read_csv("Coding3_Data.csv")
x = data["x"].values
y = data["y"].values

span_values = np.linspace(0.2, 0.9, 15)

best_span_loocv = best_span_using_loocv(x, y, span_values)
best_span_gcv = best_span_using_gcv(x, y, span_values)

print("================================================")
print(f"Best span using LOO-CV: {best_span_loocv}")
print(f"Best span using GCV: {best_span_gcv}")

# %%
best_span = best_span_loocv

loess_model = loess(x, y, span=best_span)
loess_model.fit()

x_grid = np.linspace(min(x), max(x), 100)
y_grid = loess_model.predict(x_grid)
true_y_grid = np.sin(12*(x_grid+0.2))/(x_grid+0.2)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Data points", color="blue")
plt.plot(x_grid, y_grid, label="Fitted curve", color="red")
plt.plot(x_grid, true_y_grid, label="True curve", color="green")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
if False:
    """
    Part 2.1: Ridgeless Function
    Your task is to write a function ridgeless to implement Ridgeless Least Squares, which is equivalent to
    Principal Component Regression (PCR) using all principal components, with the option scale = FALSE.
    This means that when computing the principal components (PCs), you center each column of the design
    matrix without scaling.
    • Input: Train and test datasets. In both datasets, the first column represents the response vector Y,
    and the remaining columns represent the features X.
    • Output: Training and test errors, where the error refers to the average squared error.
    """

# %%
def ridgeless(train_data, test_data):
    # Separate response and features
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]
    y_test = test_data[:, 0]
    X_test = test_data[:, 1:]

    X_train_mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - X_train_mean
    X_test_centered = X_test - X_train_mean

    b0 = np.mean(y_train)

    U, D, Vt = np.linalg.svd(X_train_centered, full_matrices=False)

    eps = 1e-10
    k = np.sum(D > eps)

    # Truncate V and update feature matrix
    V_trunc = Vt.T[:, :k]
    F = X_train_centered @ V_trunc

    # Compute LS coefficients
    alpha = (F.T @ (y_train - b0)) / (D[:k]**2)

    # Compute predictions
    y_train_pred = b0 + F @ alpha
    F_test = X_test_centered @ V_trunc
    y_test_pred = b0 + F_test @ alpha

    # Calculate MSE
    train_mse = np.mean((y_train - y_train_pred)**2)
    test_mse = np.mean((y_test - y_test_pred)**2)

    return train_mse, test_mse

# %%
if False:
    """
    Part 2.2 Simulation study
    Execute the procedure below for T = 30 times. In each iteration,
    • Randomly partition the data into training (25%) and test (75%).
    • Calculate the log of the test error from the ridgeless function using the first d columns of the data,
    where d ranges from 6 to 241. Keep in mind that the number of regression parameters spans from 5 to
    240 because the first column represents Y.
    • This will result in recording 236 test errors per iteration. These errors are the averaged mean squared
    errors based on the test data. Store those test errors in a matrix of dimensions 30-by-236.
    • Graphical display: Plot the median of the test errors (collated over the 30 iterations) in log scale against
    the count of regression parameters, which spans from 5 to 240
    """

# %%
def simulation_study(data, num_iterations=30):
    num_samples = data.shape[0]
    num_features = data.shape[1] - 1  # Subtract 1 for the response variable
    print(f"Number of features: {num_features}")
    test_errors = np.zeros((num_iterations, num_features - 4))  # 236 columns

    for iteration in tqdm(range(num_iterations)):
        train_indices = np.random.choice(num_samples, size=int(0.25 * num_samples), replace=False)
        test_indices = np.setdiff1d(np.arange(num_samples), train_indices)
        
        train_data = data.iloc[train_indices].values
        test_data = data.iloc[test_indices].values

        for d in range(6, 242):  # 6 to 241 inclusive
            subset_train = train_data[:, :d]
            subset_test = test_data[:, :d]
            
            _, test_error = ridgeless(subset_train, subset_test)
            test_errors[iteration, d-6] = np.log(test_error)

    return test_errors

data = pd.read_csv("Coding3_dataH.csv")
test_errors = simulation_study(data)
median_test_errors = np.median(test_errors, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(range(5, 241), median_test_errors)
plt.xlabel('Number of features')
plt.ylabel('Log of median test error')
plt.title('Median Test Error vs Number of Features')
plt.grid(True)
plt.show()

# %%
if False:
    """
    Part 3 Clustering time series
    Objective
    Cluster time series data based on their fluctuation patterns using natural cubic splines.
    Data
    - Download [Sales_Transactions_Dataset_Weekly.csv]. The original data is from from UCI Machine Learning Repository [Link]
    - This dataset contains the weekly purchased quantities of 811 products over 52 weeks, resulting in a time
    series of 52 data points for each product.
    - Center each time series by removing its mean, and store the resulting data in an 811-by-52 matrix X.
    """
# %%
data = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
X = data.iloc[:, 1:53].values
X = X - X.mean(axis=1, keepdims=True)

# %%

if False:
    """
    Part 3.1 Task 1: Fitting NCS
    • Fit each time series with an NCS with df = 10. This corresponds to an NCS with 8 interior knots.
    • Each row of X811x52 represents the response, with the 1-dimensional feature being the index from 1 to 52.
    • Store the NCS coefficients (excluding the intercept) in an 811-by-9 matrix B811x9.
    • Matrix B can be derived as follows:
        • F is a 52-by-9 design matrix without the intercept. For instance, this can be obtained using the
        ns command with df = 9 and intercept = FALSE.
        • Remove the column mean from F to disregard the intercept.
        • Then compute B via
        Bt = (FtF)-1FtXt.
        • The formula above is given for the transpose of B, since it is equivalent to fitting 811 linear
        regression models: the design matrix stays the same (i.e., F) but the response vector — there are
        811 response vectors corresponding to the 811 rows of X — would vary; each nine dimensional
        regression coefficient vector corresponds to a row in B (or equivalently, column in Bt).
        • If you do not want to remove the column mean from F, add the intercept column to F and denote
        the resulting 52-by-10 matrix as ˜F. Then compute the 811-by-10 matrix ˜B:
        • ˜Bt = ( ˜Ft ˜F)-1 ˜FtXt.
        • Next, obtain B by dropping the first column of ˜B, as this column corresponds to the intercepts
        from the 811 regression models.
    """

# %%
def ns(x, df=None, knots=None, boundary_knots=None, include_intercept=False):
    degree = 3
    
    if boundary_knots is None:
        boundary_knots = [np.min(x), np.max(x)]
    else:
        boundary_knots = np.sort(boundary_knots).tolist()

    oleft = x < boundary_knots[0]
    oright = x > boundary_knots[1]
    outside = oleft | oright
    inside = ~outside

    if df is not None:
        nIknots = df - 1 - include_intercept
        if nIknots < 0:
            nIknots = 0
            
        if nIknots > 0:
            knots = np.linspace(0, 1, num=nIknots + 2)[1:-1]
            knots = np.quantile(x[~outside], knots)

    Aknots = np.sort(np.concatenate((boundary_knots * 4, knots)))
    n_bases = len(Aknots) - (degree + 1)

    if any(outside):
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        e = 1 / 4 # in theory anything in (0, 1); was (implicitly) 0 in R <= 3.2.2

        if any(oleft):
            k_pivot = boundary_knots[0]
            xl = x[oleft] - k_pivot
            xl = np.c_[np.ones(xl.shape[0]), xl]

            # equivalent to splineDesign(Aknots, rep(k.pivot, ord), ord, derivs)
            tt = np.empty((xl.shape[1], n_bases), dtype=float)
            for j in range(xl.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)

            basis[oleft, :] = xl @ tt

        if any(oright):
            k_pivot = boundary_knots[1]
            xr = x[oright] - k_pivot
            xr = np.c_[np.ones(xr.shape[0]), xr]

            tt = np.empty((xr.shape[1], n_bases), dtype=float)
            for j in range(xr.shape[1]):
                for i in range(n_bases):
                    coefs = np.zeros((n_bases,))
                    coefs[i] = 1
                    tt[j, i] = splev(k_pivot, (Aknots, coefs, degree), der=j)
                    
            basis[oright, :] = xr @ tt
        
        if any(inside):
            xi = x[inside]
            tt = np.empty((len(xi), n_bases), dtype=float)
            for i in range(n_bases):
                coefs = np.zeros((n_bases,))
                coefs[i] = 1
                tt[:, i] = splev(xi, (Aknots, coefs, degree))

            basis[inside, :] = tt
    else:
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (Aknots, coefs, degree))

    const = np.empty((2, n_bases), dtype=float)
    for i in range(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        const[:, i] = splev(boundary_knots, (Aknots, coefs, degree), der=2)

    if include_intercept is False:
        basis = basis[:, 1:]
        const = const[:, 1:]

    qr_const = np.linalg.qr(const.T, mode='complete')[0]
    basis = (qr_const.T @ basis.T).T[:, 2:]

    return basis

# %%
def fit_ncs(X):
    n_features = X.shape[1]
    df = 10
    knots = np.linspace(0, 1, num=df-1)[1:-1]
    knots = np.quantile(np.arange(n_features), knots)
    knots = np.sort(np.concatenate(([0], knots, [n_features-1])))
    F = ns(np.arange(n_features), df=df, knots=knots, include_intercept=False)
    B = np.linalg.inv(F.T @ F) @ F.T @ X.T
    return B.T

# %%
if False:
    """
    Part 3.2 Cluster Matrix B
    • Run the k-means algorithm on matrix B to cluster the 811 products into 6 clusters.
    • Visualize the centered time series (i.e., rows of X), colored in grey, of products grouped by their
    clusters. Overlay the plots with the cluster centers (colored in red). Arrange the visualizations in a
    2-by-3 format.
    • Note: When using matrix B for clustering, the centers are the average of the rows of B within each
    cluster. To get the corresponding time series from a cluster center b, use the matrix product Fb.
    """

# %%
B = fit_ncs(X)
kmeans = KMeans(n_clusters=6, random_state=0)
clusters = kmeans.fit_predict(B)
cluster_centers = kmeans.cluster_centers_

# Create the design matrix F
n_features = X.shape[1]
df = 10
knots = np.linspace(0, 1, num=df-1)[1:-1]
knots = np.quantile(np.arange(n_features), knots)
knots = np.sort(np.concatenate(([0], knots, [n_features-1])))
F = ns(np.arange(n_features), df=df, knots=knots, include_intercept=False)

center_time_series = F @ cluster_centers.T

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()

for i in range(6):
    cluster_mask = clusters == i
    axs[i].plot(X[cluster_mask].T, color='grey', alpha=0.1)
    axs[i].plot(center_time_series[:, i], color='red', linewidth=2)
    axs[i].set_title(f'Cluster {i+1}')
    axs[i].set_xlabel('Week')
    axs[i].set_ylim(-20, 30)
    axs[i].set_ylabel('Centered Sales')

plt.tight_layout()
plt.show()


# %%
if False:
    """
    Part 3.3 Cluster Matrix X
    • Run the k-means algorithm on matrix X to cluster the 811 products into 6 clusters.
    • Similarly, visualize the centered time series of products grouped by their clusters, accompanied by
    their respective cluster centers. Arrange the visualizations in a 2-by-3 format
    """

# %%
kmeans_X = KMeans(n_clusters=6, random_state=0)
clusters_X = kmeans_X.fit_predict(X)
cluster_centers_X = kmeans_X.cluster_centers_

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()

for i in range(6):
    cluster_mask = clusters_X == i
    axs[i].plot(X[cluster_mask].T, color='grey', alpha=0.1)
    axs[i].plot(cluster_centers_X[i], color='red', linewidth=2)
    axs[i].set_title(f'Cluster {i+1}')
    axs[i].set_xlabel('Week')
    axs[i].set_ylim(-20, 30)
    axs[i].set_ylabel('Centered Sales')

plt.tight_layout()
plt.show()

# %%
