import numpy as np
from sklearn.linear_model import Lasso
from scipy.linalg import cho_factor, cho_solve, cholesky
import time
# from ray.util.multiprocessing import Pool
from multiprocessing import Pool
from lasso import lasso_input_check, shrinkage
from main import *


# Objective function
def objective(A, y, beta, lam):
    return 0.5 * np.linalg.norm(y - np.dot(A, beta)) ** 2 + lam * np.linalg.norm(beta, 1)

def admm_lasso_multi(A, b, lam=1.0, rho=1.0, max_iter=1000, tol=1e-4, verbose=False, return_history=False, num_procs=4):
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
    uhat = u
    xhat = x

    if return_history:
        objs = np.zeros((max_iter,))
        r_norms = np.zeros((max_iter,))
        s_norms = np.zeros((max_iter,))
        eps_pris = np.zeros((max_iter,))
        eps_duals = np.zeros((max_iter,))

    # Start timer
    t1 = time.time()
    # Precompute constants
    n_features_sqrt_abstol = np.sqrt(n_features) * ABSTOL

    L_multi = {}
    U_multi = {}
    u_multi = {}
    x_multi = {}
    A_multi = np.split(A, num_procs, axis=0)
    b_multi = np.split(b, num_procs, axis=0)

    def cholesky_multi(i):
        L_i = cholesky(np.dot(A_multi[i].T, A_multi[i]) + rho * np.eye(n_features), lower=True)
        U_i = L_i.T
        u_i = np.zeros((n_features,))
        return L_i, U_i, u_i

    def solve_multi(i):
        return np.linalg.solve(U_multi[i], np.linalg.solve(L_multi[i], np.dot(A_multi[i].T, b_multi[i]) + rho * (z - u)))

    def update_u(i):
        return u_multi[i] + (x_multi[i] - z)

    with Pool(processes=num_procs) as pool:
        for i in range(max_iter):
            if i == 0:
                for i, (L_i, U_i, u_i) in enumerate(pool.map(cholesky_multi, range(num_procs))):
                    L_multi[i] = L_i
                    U_multi[i] = U_i
                    u_multi[i] = u_i

            # Update x
            for i, x_i in enumerate(pool.map(solve_multi, range(num_procs))):
                x_multi[i] = x_i

            for i in range(num_procs):
                xhat += x_multi[i]
            xhat /= num_procs

            # Update z with relaxation
            z = shrinkage(xhat + uhat, lam / (rho * num_procs))
            z_old = z

            for i, u_i in enumerate(pool.map(update_u, range(num_procs))):
                u_multi[i] = u_i


            for i in range(num_procs):
                uhat += u_multi[i]
            uhat /= num_procs

            # Check convergence
            r_norm = np.linalg.norm(xhat - z, 2)
            s_norm = np.linalg.norm(-rho * (z - z_old), 2)

            eps_pri = n_features_sqrt_abstol + RELTOL * max(np.linalg.norm(xhat, 2), np.linalg.norm(-z, 2))
            eps_dual = n_features_sqrt_abstol + RELTOL * np.linalg.norm(rho * uhat, 2)

            pri_converged = r_norm < eps_pri
            dual_converged = s_norm < eps_dual

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

if __name__ == "__main__":
    data = sample_problem_2()
    A, x0, b, lam, rho, max_iter, tol = data

    print("Testing lasso")
    print("=============")
    print("Running with n = {}, p = {}, lam = {}, rho = {}".format(A.shape[0], A.shape[1], lam, rho))

    x_new, stats_new = admm_lasso(A, b, lam, rho, max_iter, tol)
    # x, stats = lasso(A, b, lam, rho, max_iter, tol)
    x_lib, stats_lib = lasso_lib(A, b, lam, rho, max_iter, tol)

    x_multi, stats_multi = admm_lasso_multi(A, b, lam, rho, max_iter, tol, False, False, 4)

    # Check convergence
    print("Checking convergence")
    print("lasso error: {}".format(compute_error(x_new, x_lib)))

    print("lasso iterations: {}".format(stats_new["iter"]))
    print("lasso time: {}".format(stats_new["time"]))
    print("lasso_new objective: {}".format(stats_new["objective"][-1]))

    print("lasso_lib iterations: {}".format(stats_lib["iter"]))
    print("lasso_lib time: {}".format(stats_lib["time"]))
    print("lasso_lib objective: {}".format(stats_lib["objective"][-1]))

    print("lasso_lib iterations: {}".format(stats_lib["iter"]))
    print("lasso_lib time: {}".format(stats_lib["time"]))
    print("lasso_lib objective: {}".format(stats_lib["objective"][-1]))

    print("lasso vs. multi lasso error: {}".format(compute_error(x_new, x_multi)))
    print("lasso multi time: {}".format(stats_lib["time"]))

    print("Done!")
