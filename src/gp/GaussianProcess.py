import numpy as np
import scipy
import scipy.linalg

from src.gp.covariance.CovarianceGenerator import CovarianceGenerator
from src.utils.Util import covariance_mesh


class GaussianProcess:
    def __init__(self, hyper_storer, mean_function, noise_variance):
        covariance_generator = CovarianceGenerator(hyper_storer)
        self.covariance_function = covariance_generator.get_covariance().cov
        self.mean_function = mean_function
        self.noise_variance = noise_variance

    # assert locations, current_location a 2-D arrays
    def _mean(self, k_star, cholesky, measurements):
        # Weights by matrix division using cholesky decomposition
        weights = scipy.linalg.cho_solve((cholesky, True), k_star).T
        mean = np.dot(weights, measurements - np.ones(measurements.shape) * self.mean_function) + self.mean_function

        return mean

    # assert locations, current_location a 2-D arrays
    def cholesky(self, locations):
        k = covariance_mesh(locations, locations, self.covariance_function)
        return np.linalg.cholesky(k + self.noise_variance * np.identity(k.shape[0]))

    def _variance(self, k_star, current_location, cholesky, include_noise_in_variance):
        # include_noise_in_variance parameter:
        # if we calculate posterior p(y | x) we need to add \sigma_n^2
        # if we calculate posterior p(f | x) we do not need to add \sigma_n^2

        assert cholesky is not None
        # k_star = self.__covariance_mesh(locations, current_location)
        k_current = covariance_mesh(current_location, current_location, self.covariance_function)
        v = scipy.linalg.solve_triangular(cholesky, k_star, lower=True)
        v_product = np.dot(v.T, v)
        assert v_product.shape == k_current.shape
        var = k_current - v_product

        if include_noise_in_variance:
            var += self.noise_variance * np.identity(k_current.shape[0])
            
        return var

    def predict(self, locations, measurements, current_location, cholesky=None, include_noise_in_variance=False):
        if cholesky is None:
            cholesky = self.cholesky(locations)
        k_star = covariance_mesh(locations, current_location, self.covariance_function)
        var = self._variance(k_star=k_star,
                             current_location=current_location,
                             cholesky=cholesky,
                             include_noise_in_variance=include_noise_in_variance)
        mean = self._mean(k_star=k_star, cholesky=cholesky,
                          measurements=measurements)
        return mean, var


if __name__ == "__main__":
    # Generation Tests
    pass
