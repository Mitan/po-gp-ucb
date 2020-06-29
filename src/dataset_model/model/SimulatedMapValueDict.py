from functools import reduce

from src.enum.DatasetEnum import DatasetEnum
from src.dataset_model.model.MapValueDictBase import MapValueDictBase
import numpy as np

from scipy.stats import multivariate_normal

from src.gp.covariance.CovarianceGenerator import CovarianceGenerator
from src.utils.Util import get_full_dataset_covariance_mesh


class SimulatedMapValueDict(MapValueDictBase):

    def __init__(self, hyper_storer, domain_descriptor, filename=None):
        self.dataset_type = DatasetEnum.Simulated
        self.hyper_storer = hyper_storer
        self.domain_descriptor = domain_descriptor
        if filename:
            data = np.genfromtxt(filename, delimiter=',')
            locs = data[:, :-1]
            vals = data[:, -1]
            data_point_dimension = locs.shape[1]
            assert data_point_dimension == self.domain_descriptor.point_dimension,\
                "domain descriptor point dimension conflicts with  data point dimension"
        else:
            covariance_generator = CovarianceGenerator(hyper_storer)
            covariance = covariance_generator.get_covariance()
            locs, vals = self.__generate_values(covariance=covariance,
                                                grid_domain=domain_descriptor.grid_domain,
                                                num_samples=domain_descriptor.num_samples_grid,
                                                noise_variance=hyper_storer.noise_variance)

        self.point_dimension = self.domain_descriptor.point_dimension

        MapValueDictBase.__init__(self, locations=locs, values=vals)

    # generates values with zero mean
    @staticmethod
    def __generate_values(covariance, grid_domain, num_samples, noise_variance):

        assert (len(grid_domain) == len(num_samples))

        # Number of dimensions of the multivariate gaussian is equal to the number of grid points
        ndims = len(num_samples)
        # todo note end points are included, so subtract 1
        grid_res = [float(grid_domain[x][1] - grid_domain[x][0]) / float(num_samples[x] - 1) for x in range(ndims)]
        npoints = reduce(lambda a, b: a * b, num_samples)

        # Mean function is assumed to be zero
        u = np.zeros(npoints)

        # List of points
        epsilon_upper_end = 10**(-8)
        grid1dim = [slice(grid_domain[x][0], grid_domain[x][1] + epsilon_upper_end, grid_res[x]) for x in range(ndims)]
        grids = np.mgrid[grid1dim]
        points = grids.reshape(ndims, -1).T

        assert points.shape[0] == npoints

        # construct covariance matrix
        cov_mat = get_full_dataset_covariance_mesh(points, covariance)

        # Draw vector
        # previously used the seed variable as the seed
        np.random.seed()
        # these are noiseless observations
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        # add noise to them
        noise_components = np.random.normal(0, np.math.sqrt(noise_variance), npoints)
        assert drawn_vector.shape == noise_components.shape
        assert drawn_vector.shape[0] == npoints
        drawn_vector_with_noise = np.add(drawn_vector, noise_components)

        return points, drawn_vector_with_noise
