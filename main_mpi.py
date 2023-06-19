import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from lasso import admm_lasso, admm_lasso_mpi, lasso_lib, generate_lasso_data, tol_check, compute_error
from utils import get_file_path, create_output_dir

import mpi4py
mpi4py.rc.threads = False
from mpi4py import MPI

# Generate data
def sample_problem_0():
    """ Sample problem 1 - Small data set"""
    n = 10
    p = 5
    sigma = 0.1
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
    sigma = 0.1
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
    sigma = 0.1
    np.random.seed(0)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 1e-7 # 0.1 * np.linalg.norm(np.dot(A.T, b), ord=np.inf)
    rho = 1
    max_iter = 1000
    tol = 1e-9
    return A, x0, b, lam, rho, max_iter, tol

def sample_problem_3():
    n = 100000
    p = 50
    sigma = 0.1
    np.random.seed(0)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 1e-5
    rho = 1
    max_iter = 1000
    tol = 1e-12
    return A, x0, b, lam, rho, max_iter, tol


def sample_problem_0_mpi(seed=0):
    """ Sample problem 0 - Tiny data set"""
    n = 10
    p = 5
    sigma = 0.1
    # Set seed
    np.random.seed(seed)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 0.01
    rho = 1
    max_iter = 1000
    tol = 1e-6
    return A, x0, b, lam, rho, max_iter, tol

def sample_problem_1_mpi(seed=0):
    """ Sample problem 1 - Small data set"""
    n = 100000
    p = 20
    sigma = 0.1
    # Set seed
    np.random.seed(seed)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 0.01
    rho = 1
    max_iter = 1000
    tol = 1e-6
    return A, x0, b, lam, rho, max_iter, tol

def sample_problem_3_mpi(seed=0):
    n = 1000000
    p = 50
    sigma = 0.1
    np.random.seed(seed)
    A, x0, b = generate_lasso_data(n, p, sigma)
    lam = 1e-5
    rho = 1
    max_iter = 1000
    tol = 1e-12
    return A, x0, b, lam, rho, max_iter, tol


if __name__ == "__main__":

    # Run MPI test
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print("Hello from rank", rank)

    data = sample_problem_1_mpi(seed=rank)
    A, x0, b, lam, rho, max_iter, tol = data

    if rank == 0:
        print("Testing lasso MPI")
        print("Running on {} cores".format(size))
        print("=============")
        print("Running with n = {}, p = {}, lam = {}, rho = {}".format(A.shape[0], A.shape[1], lam, rho))
        print("Running with max_iter = {}, tol = {}".format(max_iter, tol))
        print("=============")
    
    comm.Barrier()
    x_new, stats_new = admm_lasso_mpi(A, b, lam, rho, max_iter, tol, verbose=False, return_history=False, comm=comm, debug=True)
    
    # Check convergence
    if rank == 0:
        print("Checking convergence")
    print("rank {} lasso error: {}".format(rank, compute_error(x_new, x0)))
    print("rank {} lasso iterations: {}".format(rank, stats_new['iter']))
    # print first 6 x values
    print("rank {} x_new[:6] = {}".format(rank, x_new[:6]))

    