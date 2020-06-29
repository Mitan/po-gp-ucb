import logging
import math
from math import sqrt

import numpy as np

from src.enum.MethodEnum import MethodEnum
from src.gp.covariance.SquareExponential import SquareExponential
from src.utils.FileUtil import write_results_to_file

log = logging.getLogger(__name__)


# here regrets are instant regrets at each iteration. so it's just max_value - current-measurement
def get_means_and_variances(regrets):
    num_points, num_iterations = regrets.shape
    # calculate min regret at each step for a given seed
    for i in range(num_points):
        for j in range(1, num_iterations):
            regrets[i][j] = min(regrets[i][j], regrets[i][j - 1])

    mean = np.mean(regrets, axis=0)
    var = np.std(regrets, axis=0) / sqrt(num_points)
    return mean, var


def covariance_mesh(col, row, covariance_function):
    cols = col.shape[0]
    rows = row.shape[0]
    cov_mat = np.zeros((cols, rows), float)
    for y in range(cols):
        for x in range(rows):
            cov_mat[y, x] = covariance_function(row[x, :], col[y, :])
    return cov_mat


# fast method specifically for non-ard square exponential
def _get_fast_rbf_dataset_covariance_matrix(points, covariance):
    x = np.matrix(points)
    # get a matrix where the (i, j)th element is |x[i] - x[j]|^2
    # using the identity (x - y)^T (x - y) = x^T x + y^T y - 2 x^T y
    pt_sq_norms = np.square(x).sum(axis=1)
    dists_sq = np.dot(x, x.T)
    dists_sq *= -2
    dists_sq += pt_sq_norms.reshape(1, -1)
    dists_sq += pt_sq_norms

    km = dists_sq

    km /= -2 * covariance.length_scale ** 2
    np.exp(km, km)
    km *= covariance.signal_variance
    return km


def get_full_dataset_covariance_mesh(points, covariance):
    if type(covariance) == SquareExponential:
        log.info("Using fast covariance matrix generation for SquareExponential non-ARD kernel")
        return _get_fast_rbf_dataset_covariance_matrix(points, covariance)
    return covariance_mesh(points, points, covariance.cov)


def process_results(regrets, results_filename):
    means, error_bars = get_means_and_variances(regrets=regrets)
    write_results_to_file(means=means,
                          variances=error_bars,
                          filename=results_filename)


# for some dataset BO is done on transformed data, e.g. on log-GP.
# For plotting we need to transform it back
def transform_results_back(value, dataset_type):
    return value

    # if dataset_type == DatasetEnum.Simulated:
    #     return value
    # # see preprocessing dataset script for the meaning of these values
    # elif dataset_type == DatasetEnum.HousePrice:
    #     var = 0.43461077417585525
    #     mean = 6.849994927030951
    #     return - (math.exp( - value * var + mean) + 300)
    # else:
    #     raise ValueError("Unknown dataset")


def get_results_filenames(run_mode, results_folder, epsilon, r):
    if run_mode == MethodEnum.UCB:
        log.info("Running non-private GP-UCB")
        seed_result_file = 'ucb_history.txt'
        results_filename = "{}ucb_results.txt".format(results_folder)

    elif run_mode == MethodEnum.ODP_GP_UCB:

        log.info("Running ODP-GP-UCB")
        seed_result_file = 'odp_gp_ucb_history_eps{:2.1f}_r{}.txt'.format(math.log(epsilon), r)
        results_filename = "{}odp_gp_ucb_results_e{:2.1f}_r{}.txt".format(results_folder,
                                                                   math.log(epsilon),
                                                                   r)
    else:
        raise ValueError("Unknown method")

    return seed_result_file, results_filename
