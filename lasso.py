import numpy as np
from sklearn.linear_model import Lasso
from scipy.linalg import cho_factor, cho_solve
import time


# Generate lasso data
def generate_lasso_data(n, p, sigma):
    # Generate A
    A = np.eye(n, p)  # np.random.normal(0, 1, (n, p))
    # Generate beta
    x0 = np.random.normal(0, 1, p) + 10
    # Generate y
    b = np.dot(A, x0) # + np.random.normal(0, sigma, n)
    return A, x0, b


# Generate lasso data with sparsity
def generate_lasso_data_sparse(n, p, sigma, sparsity):
    # Generate A
    A = np.random.normal(0, 1, (n, p))
    # Normalize A columns
    A = A / np.sqrt(np.sum(A ** 2))
    # Generate beta
    x0 = np.zeros(p)
    x0[:sparsity] = np.random.normal(0, 1, sparsity)
    # Generate y
    b = np.dot(A, x0) + np.random.normal(0, sigma, n)
    return A, x0, b


# Compute error compared to true beta
def compute_error(beta, beta_true):
    return np.linalg.norm(beta - beta_true, ord=2)


# Soft thresholding operator
def soft_threshold(A, kappa):
    return np.sign(A) * np.maximum(np.abs(A) - kappa, 0)


# Check convergence
def tol_check(A, y, tol):
    return np.linalg.norm(A - y) / np.linalg.norm(A) < tol


# Objective function
def objective(A, y, beta, lam):
    return 0.5 * np.linalg.norm(y - np.dot(A, beta)) ** 2 + lam * np.linalg.norm(beta, 1)


# Input check for lasso
def lasso_input_check(A, y, lam, rho, max_iter, tol):
    # Check A
    if not isinstance(A, np.ndarray):
        raise TypeError("A must be a numpy array")
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    # Check y
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if len(y) != A.shape[0]:
        raise ValueError("y must have the same length as A")
    # Check lam
    if not isinstance(lam, (int, float)):
        raise TypeError("lam must be a float")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    # Check rho
    if not isinstance(rho, (int, float)):
        raise TypeError("rho must be a float")
    if rho < 0:
        raise ValueError("rho must be non-negative")
    # Check max_iter
    if not isinstance(max_iter, int):
        raise TypeError("max_iter must be an integer")
    if max_iter < 0:
        raise ValueError("max_iter must be positive")
    # Check tol
    if not isinstance(tol, float):
        raise TypeError("tol must be a float")
    if tol < 0:
        raise ValueError("tol must be non-negative")


# ADMM for lasso problem
def lasso(A, y, lam, rho, max_iter, tol):
    # Check input
    lasso_input_check(A, y, lam, rho, max_iter, tol)
    # Get dimensions
    n, p = A.shape
    # Initialize
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    # Precompute
    XTX = np.dot(A.T, A)
    XTy = np.dot(A.T, y)
    # Collect iterations statistics
    beta_all = np.zeros((p, max_iter))
    z_all = np.zeros((p, max_iter))
    u_all = np.zeros((p, max_iter))
    obj_all = np.zeros(max_iter)

    # ADMM
    t0 = time.time()
    for i in range(max_iter):
        # Update beta
        beta = np.linalg.solve(XTX + rho * np.eye(p), XTy + rho * (z - u))
        # Update z
        zold = z
        z = soft_threshold(beta + u, lam / rho)

        # Update u
        u = u + beta - z

        # Collect iterations statistics
        beta_all[:, i] = beta
        z_all[:, i] = z
        u_all[:, i] = u
        obj_all[i] = objective(A, y, beta, lam)

        # Check convergence
        if tol_check(z, zold, tol):
            break
    te = time.time() - t0

    if i == max_iter - 1:
        print('ADMM did not converge in {} iterations'.format(max_iter))

    # Combine Statistics into a dictionary
    stats = {'beta': beta_all[:, :i], 'z': z_all[:, :i], 'u': u_all[:, :i], 'iter': i, 'time': te}

    return z, stats

def admm_lasso(A, b, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4):

    # Check input
    lasso_input_check(A, b, lam, rho, max_iter, tol)
    abs_tol = 1.0

    n_samples, n_features = A.shape

    # Initialize variables
    x = np.zeros((n_features,))
    z = np.zeros((n_features,))
    u = np.zeros((n_features,))

    # Precompute matrices
    Atb = np.dot(A.T, b)

    # Cholesky factorization of XTX + rho * I
    L, lower = cho_factor(np.dot(A.T, A) + rho * np.eye(n_features), lower=True)

    t1 = time.time()
    for i in range(max_iter):
        # Update x
        x = cho_solve((L, lower), Atb + rho * (z - u))

        # Update z with relaxation
        z = soft_threshold(x + u, lam / rho)

        # Update u
        u += x - z

        # Check convergence
        r = np.linalg.norm(x - z, 2)
        r_norm = r / np.linalg.norm(x, 1)

        # Check convergence
        if r < abs_tol and r_norm < tol:
            break
    te = time.time() - t1

    if i == max_iter - 1:
        print('ADMM Lasso did not converge in {} iterations'.format(max_iter))

    stats = {'x': x, 'z': z, 'u': u, 'iter': i+1, 'time': te}

    return x, stats

# Reference lasso solution from library
def lasso_lib(A, y, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4):
    clf = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter, tol=tol)
    t1 = time.time()
    clf.fit(A, y)
    te = time.time() - t1
    stats = {'x': clf.coef_, 'z': clf.coef_, 'u': np.zeros(len(clf.coef_)), 'iter': clf.n_iter_, 'time': te}
    return clf.coef_, stats
