import numpy as np
import matplotlib.pyplot as plt
from lasso import lasso, admm_lasso, lasso_lib, generate_lasso_data, tol_check, compute_error
from utils import get_file_path, create_output_dir

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
    # lam = 1e-5
    lam = 10
    rho = 1
    max_iter = 1000
    tol = 1e-12
    return A, x0, b, lam, rho, max_iter, tol


def plot_stats(data, stats, name=None, save=False):
    """ Plot stats """
    if len(stats["objective"]) <= 1:
        print("No stats to plot - only one iteration (try to set return_history=True)")
        return

    create_output_dir("project")

    if name is None:
        name = "lasso"
    # Plot objective value, as a function of iterations
    plt.figure()
    plt.ticklabel_format(style='plain')
    plt.plot(stats["objective"])
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.yscale("log")
    plt.suptitle("Objective value vs. iteration")
    plt.title(f"$\lambda = {data[3]}, \\rho = {data[4]}$")
    if save:
        save_path = get_file_path("{}_objective".format(name))
        plt.savefig("{}.pdf".format(save_path), bbox_inches='tight')
    else:
        plt.show()

    # Plot r_norm and s_norm, as a function of iterations
    plt.figure()
    plt.plot(stats["r_norm"], label="r_norm")
    plt.plot(stats["s_norm"], label="s_norm")
    # Add eps_duals and eps_primals
    plt.plot(stats["eps_pri"], label="eps_primal", linestyle='--')
    plt.plot(stats["eps_dual"], label="eps_dual", linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.yscale("log")
    plt.suptitle("Norms vs. iteration")
    plt.title("$\lambda = {}, \\rho = {}$".format(data[3], data[4]))
    plt.legend()
    if save:
        save_path = get_file_path("{}_norms".format(name))
        plt.savefig("{}.pdf".format(save_path), bbox_inches='tight')
    else:
        plt.show()


def run_reestimation_test(function, problem, name=None, save=False):
    """ Run reestimation test - to see if the solution is the same for different lambda values """

    data = problem()
    A, x0, b, _, rho, max_iter, tol = data

    # Run with different lambda values
    lams = np.logspace(-3, -1, 20)
    xs = np.empty((A.shape[1], len(lams)))
    norms_linf = np.empty(len(lams))
    norms_l1 = np.empty(len(lams))
    norms_l2 = np.empty(len(lams))
    print("Running reestimation test - running for {} lambda values".format(len(lams)))
    for i, lam in enumerate(lams):
        x, stats = function(A, b, lam, rho, max_iter, tol)
        xs[:, i] = x
        norms_linf[i] = np.linalg.norm(x - x0, ord=np.inf)
        norms_l1[i] = np.linalg.norm(x - x0, ord=1)
        norms_l2[i] = np.linalg.norm(x - x0, ord=2)

    # Plot results
    plt.figure()
    plt.plot(lams, norms_l1, label="$\ell_1$")
    plt.axvline(x=lams[np.argmin(norms_l1)], color='r', linestyle="--", label=f'Optimal $\lambda$: {lams[np.argmin(norms_l1)]:2f}')
    plt.xlabel("$\lambda$")
    plt.ylabel("$\|x_0 - x\|_1$")
    plt.xscale("log")
    plt.legend()
    plt.suptitle("Reestimation test - {}".format(function.__name__))
    plt.title("Problem size: m = {}, n = {}".format(A.shape[0], A.shape[1]))
    if save:
        save_path = get_file_path(f"reestimation_{function.__name__}_{name}")
        plt.savefig("{}.pdf".format(save_path), bbox_inches='tight')
    else:
        plt.show()

    # Print best lambda for each norm
    print("Best lambda for l_inf: {}".format(lams[np.argmin(norms_linf)]))
    print("Best lambda for l_1: {}".format(lams[np.argmin(norms_l1)]))
    print("Best lambda for l_2: {}".format(lams[np.argmin(norms_l2)]))


if __name__ == "__main__":
    data = sample_problem_3()
    A, x0, b, lam, rho, max_iter, tol = data

    print("Testing lasso")
    print("=============")
    print("Running with n = {}, p = {}, lam = {}, rho = {}".format(A.shape[0], A.shape[1], lam, rho))
    print("Running with max_iter = {}, tol = {}".format(max_iter, tol))

    x_new, stats_new = admm_lasso(A, b, lam, rho, max_iter, tol, verbose=False, return_history=True)
    plot_stats(data, stats_new, name=f"lambda{lam}", save=True)
    # Run reestimation test - see reestimation of x as a function of lambda
    run_reestimation_test(admm_lasso, sample_problem_3, name="lasso", save=True)

    # x, stats = lasso(A, b, lam, rho, max_iter, tol)
    x_lib, stats_lib = lasso_lib(A, b, lam, rho, max_iter, tol)
    # Run reestimation test - see reestimation of x as a function of lambda
    # run_reestimation_test(lasso_lib, sample_problem_3, save=False)

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
