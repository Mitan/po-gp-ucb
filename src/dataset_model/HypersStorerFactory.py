from src.enum.CovarianceEnum import CovarianceEnum
from src.enum.DatasetEnum import DatasetEnum

import logging

log = logging.getLogger(__name__)


# the hypers are hard-coded for the case when the max norm of the data points is 1.
# if the max norm is scaled, the lengthscale has to be scaled accordingly
def get_hyper_storer(dataset_type, dataset_max_norm=1.0):
    if dataset_type == DatasetEnum.Simulated:
        return SimulatedSEHyperStorer(dataset_max_norm=dataset_max_norm)
    elif dataset_type == DatasetEnum.HousePrice:
        return HousePriceSEHyperStorer(dataset_max_norm=dataset_max_norm)
    elif dataset_type == DatasetEnum.Loan:
        return LoanHyperStorer(dataset_max_norm=dataset_max_norm)
    elif dataset_type == DatasetEnum.Branin:
        return BraninHyperStorer(dataset_max_norm=dataset_max_norm)
    else:
        raise ValueError("Unknown dataset")


class AbstarctHypersStorer:
    def __init__(self):
        pass

    def print_params(self):
        log.info("Length scale:{}, signal variance:{}, noise variance: {}, mean function:{}".format(
            self.length_scale, self.signal_variance, self.noise_variance, self.mean_function))

    def print_params_to_file(self, file_name):
        f = open(file_name, 'w')
        f.write("mean = " + str(self.mean_function) + '\n')
        f.write("lengthscale = " + str(self.length_scale) + '\n')
        f.write("noise = " + str(self.noise_variance) + '\n')
        f.write("signal = " + str(self.signal_variance) + '\n')
        f.close()


# non ARD
class SimulatedSEHyperStorer(AbstarctHypersStorer):
    def __init__(self, dataset_max_norm):
        AbstarctHypersStorer.__init__(self)
        self.ARD = False
        self.type = CovarianceEnum.SquareExponential
        self.length_scale = 0.25 * dataset_max_norm
        self.signal_variance = 1.0
        self.noise_variance = 0.00001
        self.mean_function = 0.0


class LoanHyperStorer(AbstarctHypersStorer):
    def __init__(self, dataset_max_norm):
        AbstarctHypersStorer.__init__(self)
        self.ARD = False
        self.type = CovarianceEnum.SquareExponential
        self.length_scale = 0.210 * dataset_max_norm
        self.signal_variance = 2.118133
        self.noise_variance = 0.830346
        self.mean_function = -0.008272


class HousePriceSEHyperStorer(AbstarctHypersStorer):
    def __init__(self, dataset_max_norm):
        AbstarctHypersStorer.__init__(self)
        self.ARD = False
        self.type = CovarianceEnum.SquareExponential
        self.length_scale = 0.110839 * dataset_max_norm
        self.signal_variance = 0.545369
        self.noise_variance = 0.527140
        self.mean_function = 0.0


class BraninHyperStorer(AbstarctHypersStorer):
    def __init__(self, dataset_max_norm):
        AbstarctHypersStorer.__init__(self)
        self.ARD = False
        self.type = CovarianceEnum.SquareExponential
        # branin
        self.signal_variance = 1.04964136
        self.noise_variance = 0.1
        self.length_scale = 0.265813188130217 * dataset_max_norm
        self.mean_function = -4.26611262226847





