# %% (PSL) Coding Assignment 4
# Chaojie Wang (netID: 656449601)
# UIUC MCS Online Fall 2024

# %% Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# %% Set the seed
np.random.seed(9601)

# %%
if False:
    """
    Part 1: Gaussian Mixtures
    
    Part 1.1: Implement the EM algorithm from scratch for a p-dimensional Gaussian mixture model with G components

    Your implementation should consists of four functions.
    • Estep function: This function should return an n-by-G matrix, where the (i, j)th entry represents the
    conditional probability P (Zi = k | xi). Here i ranges from 1 to n and k ranges from 1 to G.
    • Mstep function: This function should return the updated parameters for the Gaussian mixture model.
    • loglik function: This function computes the log-likelihood of the data given the parameters.
    • myEM function (main function): Inside this function, you can call the Estep and Mstep functions. The function should take the following inputs and return the estimated parameters and log-likelihood (via the loglik function):
        Input:
        -  data: The dataset.
        -  G: The number of components. Although your code will be tested with G = 2 and G = 3, it should be able to handle any value of G. (You can, of course, ignore the case where G > n.)
        -  Initial parameters.
        -  itmax: The number of iterations.
        Output:
        -  prob: A G-dimensional probability vector (p1, . . . , pG)
        -  mean: A p-by-G matrix with the k-th column being μk, the p-dimensional mean for the k-th Gaussian component.
        -  Sigma: A p-by-p covariance matrix Σ shared by all G components
        -  loglik: A number equal to the log-likelihood of the data given the estimated parameters.
    
    Implementation Guidelines:
    The requirements are very similar to Coding Assignment 1. “No loops” means no explicit loops such as for or while, and no use of functions like apply or map.
    • Estep function: No loops.
    • Mstep function: You may only loop over G when updating Sigma.
    • loglik function: You may only loop over G.
    • You are not allowed to use pre-existing functions or packages for evaluating normal densities. However,
    you may use built-in functions to compute the inverse of a matrix or perform SVD.
    """

# %%
def Estep(data, prob, mean, Sigma):
    """
    Compute the E-step of the EM algorithm
    
    Args:
        data: (n x p) array of observations
        prob: (G,) array of mixture probabilities
        mean: (p x G) array of component means
        Sigma: (p x p) shared covariance matrix
    
    Returns:
        resp: (n x G) array of responsibilities
    """
    n = data.shape[0]
    G = prob.shape[0]
    p = data.shape[1]
    
    Sigma_inv = np.linalg.inv(Sigma)
    Sigma_det = np.linalg.det(Sigma)
    
    x_centered = data.reshape(n, 1, p) - mean.T.reshape(1, G, p)
    
    quad = -0.5 * np.sum(
        np.matmul(x_centered, Sigma_inv) * x_centered,
        axis=2
    )
    
    log_density = quad - 0.5 * (p * np.log(2 * np.pi) + np.log(Sigma_det))
    log_resp = log_density + np.log(prob)
    
    log_resp_max = np.max(log_resp, axis=1, keepdims=True)
    resp = np.exp(log_resp - log_resp_max)
    normalizer = np.sum(resp, axis=1, keepdims=True)
    resp = resp / normalizer
    
    return resp

def Mstep(data, resp):
    """
    Compute the M-step of the EM algorithm
    
    Args:
        data: (n x p) array of observations
        resp: (n x G) array of responsibilities
    
    Returns:
        prob: (G,) array of updated mixture probabilities
        mean: (p x G) array of updated component means
        Sigma: (p x p) updated shared covariance matrix
    """
    n = data.shape[0]
    G = resp.shape[1]
    
    prob = np.mean(resp, axis=0)
    
    mean = (data.T @ resp) / (n * prob)
    
    Sigma = np.zeros_like(data.T @ data)
    for k in range(G):
        x_centered = data - mean[:, k].T
        Sigma += (resp[:, k].reshape(-1, 1) * x_centered).T @ x_centered
    Sigma /= n
    
    return prob, mean, Sigma

def loglik(data, prob, mean, Sigma):
    """
    Compute the log-likelihood of the data
    
    Args:
        data: (n x p) array of observations
        prob: (G,) array of mixture probabilities
        mean: (p x G) array of component means
        Sigma: (p x p) shared covariance matrix
    
    Returns:
        ll: log-likelihood value = sum_i log(sum_k p_k * N(x_i; mu_k, Σ))
    """
    n = data.shape[0]
    G = prob.shape[0]
    p = data.shape[1]
    
    const = -0.5 * p * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(Sigma))
    Sigma_inv = np.linalg.inv(Sigma)
    
    mixture_densities = np.zeros(n)
    
    for k in range(G):
        x_centered = data - mean[:, k].T
        quad = np.sum(np.dot(x_centered, Sigma_inv) * x_centered, axis=1)
        density = np.exp(const - 0.5 * quad)
        mixture_densities += prob[k] * density
    
    ll = np.sum(np.log(mixture_densities))
    
    return ll

def myEM(data, G, prob_init, mean_init, Sigma_init, itmax):
    """
    Main EM algorithm function
    
    Args:
        data: (n x p) array of observations
        G: number of components
        prob_init: (G,) array of initial mixture probabilities
        mean_init: (p x G) array of initial component means
        Sigma_init: (p x p) initial shared covariance matrix
        itmax: maximum number of iterations
    
    Returns:
        prob: final mixture probabilities
        mean: final component means
        Sigma: final shared covariance matrix
        ll: final log-likelihood
    """
    prob = prob_init
    mean = mean_init
    Sigma = Sigma_init
    
    for _ in range(itmax):
        # E-step
        resp = Estep(data, prob, mean, Sigma)
        
        # M-step
        prob, mean, Sigma = Mstep(data, resp)
        
        # Compute log-likelihood
        ll = loglik(data, prob, mean, Sigma)
    
    return prob, mean, Sigma, ll

# %%
if False:
    """
    Part 1.2: Test your code with the provided dataset, [faithful.dat], with both G = 2 and G = 3.
    Part 1.2.1:For the case when G = 2, set your initial values as follows:
    • p1 = 10/n, p2 = 1 - p1.
    • μ1 = the mean of the first 10 samples; μ2 = the mean of the remaining samples.
    • Calculate Σ as 1/n(sum_i=1^10 (x_i - μ_1)(x_i - μ_1)^T + sum_i=11^n (x_i - μ_2)(x_i - μ_2)^T), Here (xi - μi) is a 2-by-1 vector and the superscript t denotes the transpose. so the resulting Σ matrix is a 2-by-2 matrix.
    Run your EM implementation with 20 iterations. Your results from myEM are expected to look like the
    following. (Even though the algorithm has not yet reached convergence, matching the expected results below
    serves as a validation that your code is functioning as intended.)
    """

# %%
faithful = pd.read_csv('faithful.dat', delim_whitespace=True)
data = faithful.values
n = data.shape[0]

G = 2
prob_init = np.array([10/n, 1 - 10/n])

mean_init = np.vstack([
    data[:10].mean(axis=0),
    data[10:].mean(axis=0)
]).T

Sigma_init = np.zeros((2, 2))
x_centered = data[:10] - mean_init[:, 0]
Sigma_init += x_centered.T @ x_centered
x_centered = data[10:] - mean_init[:, 1]
Sigma_init += x_centered.T @ x_centered
Sigma_init /= n

prob_final, mean_final, Sigma_final, ll_final = myEM(
    data, G, prob_init, mean_init, Sigma_init, itmax=20
)

print("Final parameters after 20 iterations:")
print("\nMixture probabilities:")
print(prob_final)
print("\nComponent means:")
print(mean_final)
print("\nShared covariance matrix:")
print(Sigma_final)
print("\nFinal log-likelihood:")
print(ll_final)

# %%
if False:
    """
    Part 1.2.2: For the case when G = 3, set your initial values as follows:
    • p1 = 10/n, p2 = 20/n, p3 = 1 - p1 - p2
    • μ1 = 1/10 * sum_i=1^10 xi, the mean of the first 10 samples; 
    • μ2 = 1/20 * sum_i=11^30 xi, the mean of next 20 samples; 
    • μ3 = 1/30 * sum_i=31^n xi, the mean of the remaining samples.
    • Calculate Σ as 1/n(sum_i=1^10 (x_i - μ_1)(x_i - μ_1)^T + sum_i=11^30 (x_i - μ_2)(x_i - μ_2)^T + sum_i=31^n (x_i - μ_3)(x_i - μ_3)^T)
    Run your EM implementation with 20 iterations.
    """

# %%
G = 3
prob_init = np.array([10/n, 20/n, 1 - 10/n - 20/n])

mean_init = np.vstack([
    data[:10].mean(axis=0),
    data[10:30].mean(axis=0),
    data[30:].mean(axis=0)
]).T

Sigma_init = np.zeros((2, 2))
x_centered = data[:10] - mean_init[:, 0]
Sigma_init += x_centered.T @ x_centered
x_centered = data[10:30] - mean_init[:, 1]
Sigma_init += x_centered.T @ x_centered
x_centered = data[30:] - mean_init[:, 2]
Sigma_init += x_centered.T @ x_centered
Sigma_init /= n

prob_final, mean_final, Sigma_final, ll_final = myEM(
    data, G, prob_init, mean_init, Sigma_init, itmax=20
)

print("Final parameters after 20 iterations (G=3):")
print("\nMixture probabilities:")
print(prob_final)
print("\nComponent means:")
print(mean_final)
print("\nShared covariance matrix:")
print(Sigma_final)
print("\nFinal log-likelihood:")
print(ll_final)

# %%
if False:
    """
    Part 2: HMM
    Implement the Baum-Welch (i.e., EM) algorithm and the Viterbi algorithm from scratch for a Hidden
    Markov Model (HMM) that produces an outcome sequence of discrete random variables with three distinct
    values.
    A quick review on parameters for Discrete HMM:
    • mx: Count of distinct values X can take.
    • mz: Count of distinct values Z can take.
    • w: An mz-by-1 probability vector representing the initial distribution for Z1.
    • A: The mz-by-mz transition probability matrix that models the progression from Zt to Zt+1.
    • B: The mz-by-mx emission probability matrix, indicating how X is produced from Z.
    Focus on updating the parameters A and B in your algorithm. The value for mx is given and you’ll specify mz.
    For w, initiate it uniformly but refrain from updating it within your code. The reason for this is that w
    denotes the distribution of Z1 and we only have a single sample. It’s analogous to estimating the likelihood of
    a coin toss resulting in heads by only tossing it once. Given the scant information and the minimal influence
    on the estimation of other parameters, we can skip updating it.
    """

# %%
if False:
    """
    Part 2.1: Baum-Welch Algorihtm
    The Baum-Welch Algorihtm is the EM algorithm for the HMM. Create a function named BW_onestep
    designed to carry out the E-step and M-step. This function should then be called iteratively within myBW.
    BW_onstep:
    • Input:
        - data: a T-by-1 sequence of observations
        - Current parameter values
    • Output:
        - Updated parameters: A and B
    """

# %%
def forward(data, w, A, B):
    """
    Compute forward probabilities for HMM
    
    Args:
        data: (T,) array of observations
        w: (mz,) initial state distribution
        A: (mz x mz) transition probability matrix
        B: (mz x mx) emission probability matrix
    
    Returns:
        alpha: (T x mz) forward probabilities
    """
    T = len(data)
    mz = len(w)
    alpha = np.zeros((T, mz))
    
    alpha[0] = w * B[:, data[0]]
    alpha[0] /= np.sum(alpha[0])
    
    for t in range(1, T):
        alpha[t] = (alpha[t-1] @ A) * B[:, data[t]]
        alpha[t] /= np.sum(alpha[t])
    
    return alpha

def backward(data, A, B):
    """
    Compute backward probabilities for HMM
    
    Args:
        data: (T,) array of observations
        A: (mz x mz) transition probability matrix
        B: (mz x mx) emission probability matrix
    
    Returns:
        beta: (T x mz) backward probabilities
    """
    T = len(data)
    mz = A.shape[0]
    beta = np.zeros((T, mz))
    
    beta[-1] = np.ones(mz)
    
    for t in range(T-2, -1, -1):
        beta[t] = A @ (B[:, data[t+1]] * beta[t+1])
    
    return beta

def BW_onestep(data, w, A, B):
    """
    One step of Baum-Welch algorithm
    
    Args:
        data: (T,) array of observations
        w: (mz,) initial state distribution
        A: (mz x mz) transition probability matrix
        B: (mz x mx) emission probability matrix
    
    Returns:
        A_new: Updated transition matrix
        B_new: Updated emission matrix
    """
    T = len(data)
    mz, mx = B.shape
    
    # E-step: compute forward-backward probabilities
    alpha = forward(data, w, A, B)
    beta = backward(data, A, B)
    
    # Compute gamma_t(i,j) = P(Z_t = i, Z_{t+1} = j | x)
    xi = np.zeros((T-1, mz, mz))
    for t in range(T-1):
        # xi[t][i][j] = P(Z_t = i, Z_{t+1} = j | x)
        numerator = (alpha[t].reshape(-1, 1) * A * B[:, data[t+1]] * beta[t+1])
        xi[t] = numerator / np.sum(numerator)
    # Compute gamma_t(i) = P(Z_t = i | x) = sum_j P(Z_t = i, Z_{t+1} = j | x)
    gamma = np.sum(xi, axis=-1)
    gamma = np.vstack([gamma, alpha[-1] / np.sum(alpha[-1])])

    # M-step
    A_new = np.sum(xi, axis=0)
    A_new /= np.sum(A_new, axis=1, keepdims=True)
    
    B_new = np.zeros_like(B)
    for j in range(mx):
        mask = (data == j)
        B_new[:, j] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)
    
    return A_new, B_new

def myBW(data, w_init, A_init, B_init, itmax):
    """
    Full Baum-Welch algorithm
    
    Args:
        data: (T,) array of observations
        mx: number of possible observations
        mz: number of hidden states
        itmax: maximum number of iterations
    
    Returns:
        w: Initial state distribution
        A: Final transition matrix
        B: Final emission matrix
    """
    w = w_init
    A = A_init
    B = B_init
    
    for _ in range(itmax):
        A, B = BW_onestep(data, w, A, B)
    
    return w, A, B

# %%
if False:
    """
    Part 2.2: Viterbi Algorihtm
    This algorithm outputs the most likely latent sequence considering the data and the MLE of the parameters.
    myViterbi:
    • Input:
        - data: a T-by-1 sequence of observations
        - parameters: mx, mz, w, A and B
    • Output:
        - Z: A T-by-1 sequence where each entry is a number ranging from 1 to mz.
    Note on Calculations in Viterbi:
    Many computations in HMM are based on the product of a sequence of probabilities, resulting in extremely
    small values. At times, these values are so small that software like R or Python might interpret them as
    zeros. This poses a challenge, especially for the Viterbi algorithm, where differentiating between magnitudes
    is crucial. If truncated to zero, making such distinctions becomes impossible. Therefore, it’s advisable to
    evaluate these probabilities on a logarithmic scale in the Viterbi algorithm.
    """

# %%
def myViterbi(data, mx, mz, w, A, B):
    """
    Viterbi algorithm for finding most likely state sequence
    
    Args:
        data: (T,) array of observations
        mx: number of possible observations
        mz: number of hidden states
        w: (mz,) initial state distribution
        A: (mz x mz) transition probability matrix
        B: (mz x mx) emission probability matrix
    
    Returns:
        Z: (T,) most likely state sequence (1-based indexing)
    """
    T = len(data)
    
    log_A = np.log(A)
    log_B = np.log(B)
    log_w = np.log(w)
    
    delta = np.zeros((T, mz))
    psi = np.zeros((T, mz), dtype=int)
    
    delta[0] = log_w + log_B[:, data[0]]
    
    for t in range(1, T):
        for j in range(mz):
            temp = delta[t-1] + log_A[:, j]
            psi[t, j] = np.argmax(temp)
            delta[t, j] = np.max(temp) + log_B[j, data[t]]
    
    Z = np.zeros(T, dtype=int)
    Z[T-1] = np.argmax(delta[T-1])
    
    for t in range(T-2, -1, -1):
        Z[t] = psi[t+1, Z[t+1]]
    
    return Z + 1

# %%
if False:
    """
    Part 2.3: Testing
    Part 2.3.1: Test your code with the provided data sequence: [Coding4_part2_data.txt]. Set mz = 2 and start with
    the following initial values:
    w = [0.5, 0.5]
    A = [0.5, 0.5; 0.5, 0.5]
    B = [1/9, 3/9, 5/9; 1/6, 2/6, 3/6]
    Run your implementation with 100 iterations.
    """

# %%
data = np.loadtxt('coding4_part2_data.txt', dtype=int) - 1  # Subtract 1 for 0-based indexing

mx = 3
mz = 2
itmax = 100

w_init = np.array([0.5, 0.5])
A_init = np.array([[0.5, 0.5],
                   [0.5, 0.5]])
B_init = np.array([[1/9, 3/9, 5/9],
                   [1/6, 2/6, 3/6]])

w, A, B = myBW(data, w_init, A_init, B_init, itmax)

print("Final parameters after 100 iterations:")
print("\nTransition matrix A:")
print(A)
print("\nEmission matrix B:")
print(B)

Z = myViterbi(data, mx, mz, w, A, B)

with open('Coding4_part2_Z.txt', 'r') as file:
    true_hidden_seq = np.array([int(x) for x in file.read().split()])
sequence_matches = (true_hidden_seq == Z).all()
print("\nDecoded sequence matches the true hidden state sequence:", sequence_matches)

accuracy = np.mean(true_hidden_seq == Z)
print(f"Accuracy: {accuracy:.2%}")

# %%
if False:
    """
    Part 2.3.2: Initialize matrix B such that each entry is 1/3, and run your Baum-Welch algorithm for 20 and 100
iterations. Examine the resulting A and B matrices, and explain why you obtained these outcomes.
Based on your findings, you should understand why we cannot initialize our parameters in a way that
makes the latent states indistinguishable.
    """

# %%
w_init = np.array([0.5, 0.5])
A_init = np.array([[0.5, 0.5],
                   [0.5, 0.5]])
B_init = np.ones((2, 3)) / 3

w_20, A_20, B_20 = myBW(data, w_init, A_init, B_init, itmax=20)
w_100, A_100, B_100 = myBW(data, w_init, A_init, B_init, itmax=100)

print("Results after 20 iterations:")
print("\nTransition matrix A:")
print(A_20)
print("\nEmission matrix B:")
print(B_20)

print("\nResults after 100 iterations:")
print("\nTransition matrix A:")
print(A_100)
print("\nEmission matrix B:")
print(B_100)

# %%
if False:
    """
    Observation: Resulting A and B matrices are identical at 20 and 100 iterations.
    Reason: 
    - In the E-step of Baum-Welch:
        - Forward probabilities and backward probabilities will be identical for both states
        - This leads to identical gamma for both states
    - In the M-step:
        - When updating A and B using these gammas
        - The updates will maintain the same values for both states
        - This creates a cycle where states remain indistinguishable
    The algorithm gets stuck in a local optimum where both states are identical. Even with more iterations (20 vs 100), it cannot break out of this symmetry.

    Best practice: Always initialize HMM parameters with distinct values for different states to give the algorithm a clear starting point for learning the true underlying structure.
    """

# %%
