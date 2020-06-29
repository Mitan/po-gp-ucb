from math import sqrt

"""
def get_domain_descriptor(dataset_type):
    if dataset_type == DatasetEnum.Simulated:
        return SimulatedDomainDescriptor()
    else:
        raise ValueError("Unknown dataset")
"""


class SimulatedDomainDescriptor:

    def __init__(self, dataset_max_norm):

        # number of samples in each dimension
        # samples are postioned at uniform distance on the interval [min_value, max_value]
        # both ends are included.
        # todo note each value should be >= 2
        num_samples = 100
        self.num_samples_grid = (num_samples, num_samples)

        # todo note upper values are included
        # normalized so that norms of vectors are <= 1
        normalizer = dataset_max_norm / sqrt(2)
        min_value = -1.0
        max_value = 1.0
        self.grid_domain = ((min_value * normalizer, max_value * normalizer),
                            (min_value * normalizer, max_value * normalizer))
        # self.domain_size = 50 * 50

        self.point_dimension = 2
