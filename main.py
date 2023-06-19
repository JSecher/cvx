import numpy as np
from lasso import lasso, admm_lasso, lasso_lib, generate_lasso_data, tol_check, compute_error


# Generate data
def sample_problem_0():
    """ Sample problem 1 - Small data set"""
    n = 10
    p = 5
    sigma = 1
    # Set seed
    np.random.seed(0)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 0.01
    rho = 1
    max_iter = 1000
    tol = 1e-6
    return A, x0, b, lam, rho, max_iter, tol


def sample_problem_1():
    """ Sample problem 1 - Small data set"""
    n = 200
    p = 100
    sigma = 1
    # Set seed
    np.random.seed(0)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 0.0001
    rho = 1
    max_iter = 1000
    tol = 1e-4
    return A, x0, b, lam, rho, max_iter, tol


def sample_problem_2():
    """ Sample problem 2 - Large data set"""
    n = 2000
    p = 1000
    sigma = 1
    np.random.seed(0)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 0.001 # 0.1 * np.linalg.norm(np.dot(A.T, b), ord=np.inf)
    rho = 1
    max_iter = 1000
    tol = 1e-6
    return A, x0, b, lam, rho, max_iter, tol


if __name__ == "__main__":
    data = sample_problem_2()
    A, x0, b, lam, rho, max_iter, tol = data

    print("Testing lasso")
    print("=============")
    print("Running with n = {}, p = {}, lam = {}, rho = {}".format(A.shape[0], A.shape[1], lam, rho))

    x_new, stats_new = admm_lasso(A, b, lam, rho, max_iter, tol)
    # x, stats = lasso(A, b, lam, rho, max_iter, tol)
    x_lib, stats_lib = lasso_lib(A, b, lam, rho, max_iter, tol)

    # Check convergence
    print("Checking convergence")
    print("lasso error: {}".format(compute_error(x_new, x_lib)))
    print("lasso iterations: {}".format(stats_new["iter"]))
    print("lasso time: {}".format(stats_new["time"]))
    print("lasso_new objective: {}".format(stats_new["objective"][-1]))
    print("lasso_lib iterations: {}".format(stats_lib["iter"]))
    print("lasso_lib time: {}".format(stats_lib["time"]))
    print("lasso_lib objective: {}".format(stats_lib["objective"][-1]))
    print("Done!")
