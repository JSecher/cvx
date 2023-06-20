import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
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


def sample_problem_split_A(rank, size, features=None, samples=None,seed=0):
    """ Sample problem - Split A across ranks """
    if seed is None:
        seed = 0
    if features is None:
        n = 1000000
    else:
        n = features
    if samples is None:
        p = 50
    else:
        p = samples
    sigma = 0.1
    # Set seed
    np.random.seed(seed)
    if rank == 0:
        A, x0, b = generate_lasso_data(n, p, sigma)
        # Split A
        A_split = np.array_split(A, size, axis=0)
        # A = A_split[rank]
        b_split = np.array_split(b, size, axis=0)
        #b = b_split[rank]

        # A_splits = np.array_split(A, size, axis=0)
        # sendcounts = np.array([split.shape[0] for split in A_splits])
        # displacements = np.insert(np.cumsum(sendcounts), 0, 0)[:-1]

        # split = np.empty(sendcounts[rank], dtype=A.dtype)
        # comm.Scatterv([A, sendcounts, displacements, MPI.DOUBLE], split, root=0)

    else:
        A = None
        b = None
        x0 = None
        A_split = None
        b_split = None


    # # Send the appopiate A_split 
    # if rank == 0:
    #     A_splits = np.array_split(A, size, axis=0)
    #     sendcounts = np.array([split.shape[0] for split in A_splits])
    #     displacements = np.insert(np.cumsum(sendcounts), 0, 0)[:-1]
    # else:
    #     sendcounts = None
    #     displacements = None
    

    # Send the appopiate A_split (list of ndarrays) to the owner ranks
    A = comm.scatter(A_split, root=0)
    b = comm.scatter(b_split, root=0)
    x0 = comm.bcast(x0, root=0)

    print("rank = {}, A.shape = {}, b.shape = {}".format(rank, A.shape, b.shape))
    
    # Set parameters
    lam = 0.01
    rho = 1
    max_iter = 1000
    tol = 1e-10

    # Split A
    return A, x0, b, lam, rho, max_iter, tol, n, p

def sample_problem_seperate_data(rank, size, features=None, samples=None, seed=0):
    """ Sample problem - Split A across ranks """
    if seed is None:
        seed = 0
    if features is None:
        p = 1000000
    else:
        p = features
    if samples is None:
        n = 50
    else:
        n = samples

    sigma = 0.1
    
    # Generate x0 on rank 0 and broadcast, to have the same x0 on all ranks
    # Set seed
    np.random.seed(seed)
    if rank == 0:
        x0 = np.random.normal(0, 1, p)
    else:
        x0 = None
    x0 = comm.bcast(x0, root=0)
    
    # Each rank generates its own A and b
    np.random.seed(seed + rank)
    # Generate A
    A = np.random.normal(0, 1, (n, p))
    for col in range(p):
        A[:, col] = A[:, col] / np.linalg.norm(A[:, col])
    # Generate y
    b = np.dot(A, x0) + np.random.normal(0, sigma, n)
    
    # Set parameters
    lam = 0.01
    rho = 1
    max_iter = 1000
    tol = 1e-10

    # Split A
    return A, x0, b, lam, rho, max_iter, tol, n*size, p*size


if __name__ == "__main__":
    # Run MPI test
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        parser = argparse.ArgumentParser(description='Run lasso MPI test')
        parser.add_argument('--problem', type=int, default=0, help='Problem number')
        parser.add_argument('--samples', '-n', type=int, default=None, help='Samples number')
        parser.add_argument('--features', '-p', type=int, default=None, help='Feature number')
        parser.add_argument('--seed', type=int, default=None, help='Set seed')
        parser.add_argument('--output', type=str, default="lasso_mpi_stats", help='Output file name')
        args = parser.parse_args()
        
        problem = args.problem
        output = args.output
        samples = args.samples
        features = args.features
        seed = args.seed
        print("Running problem {}".format(problem))
        print("Output file name: {}".format(output))
        if samples is not None:
            print("Samples: {}".format(samples))
        if features is not None:
            print("Features: {}".format(features))
        if seed is not None:
            print("Seed: {}".format(seed))

    else: 
        problem = None
        output = None
        samples = None
        features = None
        seed = None

    # Broadcast arguments
    problem = comm.bcast(problem, root=0)
    output = comm.bcast(output, root=0)
    samples = comm.bcast(samples, root=0)
    features = comm.bcast(features, root=0)
    seed = comm.bcast(seed, root=0)

    if problem == 0 or problem == 1:
        data = sample_problem_split_A(rank, size, features=features, samples=samples, seed=seed)
        A, x0, b, lam, rho, max_iter, tol, n, p = data
    if problem == 2:
        data = sample_problem_seperate_data(rank, size, features=features, samples=samples, seed=seed)
        A, x0, b, lam, rho, max_iter, tol, n, p = data
    if problem == 3:
        samples = samples // size
        data = sample_problem_seperate_data(rank, size, features=features, samples=samples, seed=seed)
        A, x0, b, lam, rho, max_iter, tol, n, p = data
    if problem == 4:
        data = sample_problem_split_A(rank, size, features=features, samples=samples, seed=seed)
        A, x0, b, lam, rho, max_iter, tol, n, p = data
    else:
        print(f"Problem {problem} is Not implemented")
        exit(1)

    if rank == 0:
        print("Testing lasso MPI")
        print(f"Running problem {problem} on {size} cores")
        print("=============")
        print("Running with n = {}, p = {}, lam = {}, rho = {}".format(A.shape[0], A.shape[1], lam, rho))
        print("Running with max_iter = {}, tol = {}".format(max_iter, tol))
        print("=============")
    
    if size > 1:
        comm.Barrier()
        t_full = -MPI.Wtime()
        x_new, stats_new = admm_lasso_mpi(A, b, lam, rho, max_iter, tol, verbose=False, return_history=False, comm=comm, debug=True)
        t_full += MPI.Wtime()
        comm.Barrier()
    else:
        print("Running on 1 core")
        t_full = -MPI.Wtime()
        x_new, stats_new = admm_lasso(A, b, lam, rho, max_iter, tol, verbose=False, return_history=False)
        t_full += MPI.Wtime()
    
    # Compute error for all ranks
    error = np.linalg.norm(x_new - x0)
    if problem == 4:
        if size == 1:
            # Save x_new to file
            filename = f"{output}_xhat_ref"
            file_path = get_file_path(filename)
            np.save(file_path, x_new)
            print("Wrote x_new to {}".format(file_path))
        else:
            # Load x_new from file on rank 0
            if rank == 0:
                filename = f"{output}_xhat_ref.npy"
                file_path = get_file_path(filename)
                x_new_ref = np.load(file_path)
                print("Loaded x_new_ref from {}".format(file_path))
                # Compute error
                error = np.linalg.norm(x_new - x_new_ref)
            else:
                error = None
            # Broadcast error
            error = comm.bcast(error, root=0)
    # Compute iterations for all ranks
    iter = stats_new['iter']
    # Time for all ranks
    time = stats_new['time']

    # Combine stats from all ranks
    error_all = comm.gather(error, root=0)
    iter_all = comm.gather(iter, root=0)
    time_all = comm.gather(time, root=0)
    time_full_all = comm.gather(t_full, root=0)
    m_full = comm.gather(A.shape[0], root=0)

    # Print stats to file
    if rank == 0:
        create_output_dir("project")
        filename = f"{output}.txt"
        file_path = get_file_path(filename)
        i = 1
        write_to_file = True
        while os.path.exists(file_path):
            filename = f"{output}_{i}.txt"
            file_path = get_file_path(filename)
            i += 1
            if i > 200:
                raise Exception("Too many files with the same name")

        with open(file_path, 'w') as f:
            # Print header
            f.write("# n = {}, p = {}, lam = {}, rho = {}, ".format(n, p, lam, rho))
            f.write("max_iter = {}, tol = {}\n".format(max_iter, tol))
            f.write("rank, size, error, iter, time, time_full\n")
            for i in range(size):
                f.write("{}, {}, {:.5e}, {}, {:.5f}, {:.5f}\n".format(i, size, error_all[i], iter_all[i], time_all[i], time_full_all[i]))
        print("Wrote stats to {}".format(file_path))

        # Write to combined file
        filename = f"{output}_combined.txt"
        file_path = get_file_path(filename)

        if os.path.exists(file_path):
            mode = 'a'
        else:
            mode = 'w'

        with open(file_path, mode) as f:
            if mode == 'w':
                # Print header
                f.write("# n = {}, p = {}, lam = {}, rho = {}, ".format(n, p, lam, rho))
                f.write("max_iter = {}, tol = {}\n".format(max_iter, tol))
                f.write("size, average_m, error, iter, time, time_full\n")
            
            # Write out stats, averaged over all ranks
            f.write("{}, {}, {:.5e}, {}, {:.5f}, {:.5f}\n".format(size, np.mean(m_full), np.mean(error_all), np.max(iter_all), np.max(time_all), np.max(time_full_all)))

        print("Wrote combined stats to {}".format(file_path))            
