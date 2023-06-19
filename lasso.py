import numpy as np
from sklearn.linear_model import Lasso
from scipy.linalg import cho_factor, cho_solve, cholesky
import time
import warnings
# from mpi4py import MPI

# Generate lasso data
def generate_lasso_data(n, p, sigma):
    # Generate A
    A = np.random.normal(0, 1, (n, p))
    for col in range(p):
        A[:, col] = A[:, col] / np.linalg.norm(A[:, col])
    # Generate beta
    x0 = np.random.normal(0, 1, p)
    # Generate y
    b = np.dot(A, x0) + np.random.normal(0, sigma, n)

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


def shrinkage(x, kappa):
    return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)


# Check convergence
def tol_check(A, y, tol):
    return np.linalg.norm(A - y) / np.linalg.norm(A) < tol


# Objective function
def objective(A, x, b, lam):
    return 0.5 * np.linalg.norm(b - np.dot(A, x)) ** 2 + lam * np.linalg.norm(x, 1)


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
    # Raise warning if number of features is larger than number of samples
    if A.shape[1] > A.shape[0]:
        warnings.warn("Number of features is larger than number of samples")


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


def admm_lasso(A, b, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4, verbose=False, return_history=False):
    ABSTOL = tol
    RELTOL = tol

    # Check input
    lasso_input_check(A, b, lam, rho, max_iter, tol)
    abs_tol = 1.0

    n_samples, n_features = A.shape

    # Initialize variables
    x = np.zeros((n_features,))
    z = np.zeros((n_features,))
    u = np.zeros((n_features,))
    if return_history:
        objs = np.zeros((max_iter,))
        r_norms = np.zeros((max_iter,))
        s_norms = np.zeros((max_iter,))
        eps_pris = np.zeros((max_iter,))
        eps_duals = np.zeros((max_iter,))

    # Start timer
    t1 = time.time()
    # Precompute matrices
    Atb = np.dot(A.T, b)
    # Precompute constants
    n_features_sqrt_abstol = np.sqrt(n_features) * ABSTOL
    lam_rho = lam / rho

    # Cholesky factorization of XTX + rho * I
    # L, lower = cho_factor(np.dot(A.T, A) + rho * np.eye(n_features), lower=True)
    L = cholesky(np.dot(A.T, A) + rho * np.eye(n_features), lower=True)
    U = L.T

    for i in range(max_iter):
        # Update x
        #x = cho_solve((L, lower), Atb + rho * (z - u))
        x = np.linalg.solve(U, np.linalg.solve(L, Atb + rho * (z - u)))

        # Update z with relaxation
        # z = soft_threshold(x + u, lam / rho)
        z_old = z
        z = shrinkage(x + u, lam_rho)
        # Update u
        u += x - z

        # Check convergence
        r_norm = np.linalg.norm(x - z, 2)
        s_norm = np.linalg.norm(-rho * (z - z_old), 2)

        eps_pri = n_features_sqrt_abstol + RELTOL * max(np.linalg.norm(x, 2), np.linalg.norm(-z, 2))
        eps_dual = n_features_sqrt_abstol + RELTOL * np.linalg.norm(rho * u, 2)

        pri_converged = r_norm < eps_pri
        dual_converged = s_norm < eps_dual

#        r = np.linalg.norm(x - z, 2)
 #       r_norm = r / np.linalg.norm(x, 2)
  #      pri_converged = r < ABSTOL
   #     dual_converged = r_norm < RELTOL

        # Collect objective values
        if return_history:
            objs[i] = objective(A, x, b, lam)
            r_norms[i] = r_norm
            s_norms[i] = s_norm
            eps_pris[i] = eps_pri
            eps_duals[i] = eps_dual

        if verbose:
            print('iter: {}, r_norm: {:.3e}, s_norm: {:.3e}, eps_pri: {:.3e}, eps_dual: {:.3e}, pri_converged: {}, dual_converged: {}'.format(
                i, r_norm, s_norm, eps_pri, eps_dual, pri_converged, dual_converged))

        # Check convergence
        if pri_converged and dual_converged:
            break

    te = time.time() - t1

    if i == max_iter - 1:
        print('ADMM Lasso did not converge in {} iterations'.format(max_iter))

    if not return_history:
        objs = np.array([objective(A, x, b, lam)])
        r_norms = np.array([r_norm])
        s_norms = np.array([s_norm])
        eps_pris = np.array([eps_pri])
        eps_duals = np.array([eps_dual])
    else:
        objs = objs[:i + 1]
        r_norms = r_norms[:i + 1]
        s_norms = s_norms[:i + 1]
        eps_pris = eps_pris[:i + 1]
        eps_duals = eps_duals[:i + 1]

    stats = {'x': x, 'z': z, 'u': u, 'iter': i + 1, 'time': te,
             'objective': objs,
             'r_norm': r_norms,
             's_norm': s_norms,
             'eps_pri': eps_pris,
             'eps_dual': eps_duals
             }

    return x, stats

# def admm_lasso_mpi(A, b, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4, verbose=False, return_history=False):
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#
#     ABSTOL = tol
#     RELTOL = tol
#
#     # Check input
#     lasso_input_check(A, b, lam, rho, max_iter, tol)
#     abs_tol = 1.0
#
#     n_samples, n_features = A.shape
#
#     # Initialize variables
#     x = np.zeros((n_features,))
#     z = np.zeros((n_features,))
#     u = np.zeros((n_features,))
#     if return_history:
#         objs = np.zeros((max_iter,))
#         r_norms = np.zeros((max_iter,))
#         s_norms = np.zeros((max_iter,))
#         eps_pris = np.zeros((max_iter,))
#         eps_duals = np.zeros((max_iter,))
#
#     # Start timer
#     t1 = time.time()
#     # Precompute matrices
#     Atb = np.dot(A.T, b)
#     # Precompute constants
#     n_features_sqrt_abstol = np.sqrt(n_features) * ABSTOL
#     lam_rho = lam / rho
#
#     # Cholesky factorization of XTX + rho * I
#     # L, lower = cho_factor(np.dot(A.T, A) + rho * np.eye(n_features), lower=True)
#     L = cholesky(np.dot(A.T, A) + rho * np.eye(n_features), lower=True)
#     U = L.T
#
#     for i in range(max_iter):
#         # Update x
#         #x = cho_solve((L, lower), Atb + rho * (z - u))
#         x = np.linalg.solve(U, np.linalg.solve(L, Atb + rho * (z - u)))
#
#         MPI.re
#         # Update z with relaxation
#         # z = soft_threshold(x + u, lam / rho)
#         z_old = z
#         z = shrinkage(x + u, lam_rho)
#         # Update u
#         u += x - z
#
#         # Check convergence
#         r_norm = np.linalg.norm(x - z, 2)
#         s_norm = np.linalg.norm(-rho * (z - z_old), 2)
#
#         eps_pri = n_features_sqrt_abstol + RELTOL * max(np.linalg.norm(x, 2), np.linalg.norm(-z, 2))
#         eps_dual = n_features_sqrt_abstol + RELTOL * np.linalg.norm(rho * u, 2)
#
#         pri_converged = r_norm < eps_pri
#         dual_converged = s_norm < eps_dual
#
# #        r = np.linalg.norm(x - z, 2)
#  #       r_norm = r / np.linalg.norm(x, 2)
#   #      pri_converged = r < ABSTOL
#    #     dual_converged = r_norm < RELTOL
#
#         # Collect objective values
#         if return_history:
#             objs[i] = objective(A, x, b, lam)
#             r_norms[i] = r_norm
#             s_norms[i] = s_norm
#             eps_pris[i] = eps_pri
#             eps_duals[i] = eps_dual
#
#         if verbose:
#             print('iter: {}, r_norm: {:.3e}, s_norm: {:.3e}, eps_pri: {:.3e}, eps_dual: {:.3e}, pri_converged: {}, dual_converged: {}'.format(
#                 i, r_norm, s_norm, eps_pri, eps_dual, pri_converged, dual_converged))
#
#         # Check convergence
#         if pri_converged and dual_converged:
#             break
#
#     te = time.time() - t1
#
#     if i == max_iter - 1:
#         print('ADMM Lasso did not converge in {} iterations'.format(max_iter))
#
#     if not return_history:
#         objs = np.array([objective(A, x, b, lam)])
#         r_norms = np.array([r_norm])
#         s_norms = np.array([s_norm])
#         eps_pris = np.array([eps_pri])
#         eps_duals = np.array([eps_dual])
#     else:
#         objs = objs[:i + 1]
#         r_norms = r_norms[:i + 1]
#         s_norms = s_norms[:i + 1]
#         eps_pris = eps_pris[:i + 1]
#         eps_duals = eps_duals[:i + 1]
#
#     stats = {'x': x, 'z': z, 'u': u, 'iter': i + 1, 'time': te,
#              'objective': objs,
#              'r_norm': r_norms,
#              's_norm': s_norms,
#              'eps_pri': eps_pris,
#              'eps_dual': eps_duals
#              }
#
#     return x, stats


# Reference lasso solution from library
def lasso_lib(A, y, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4):
    clf = Lasso(alpha=lam, fit_intercept=False, max_iter=max_iter, tol=tol, copy_X=True)
    t1 = time.time()
    clf.fit(A, y)
    te = time.time() - t1
    stats = {'x': clf.coef_, 'z': clf.coef_, 'u': np.zeros(len(clf.coef_)), 'iter': clf.n_iter_, 'time': te, 'objective': np.array([objective(A, clf.coef_, y, lam)])}
    return clf.coef_, stats
