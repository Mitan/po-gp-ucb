from src.bo.AbstractBOMethod import AbstractBOMethod
import math
from math import sqrt, pi


import logging
log = logging.getLogger(__name__)


class UcbBO(AbstractBOMethod):
    DELTA_BETA = 0.1
    # DEFAULT_CONSTANT_BETA = 2.0
    # DEFAULT_IS_CONSTANT_BETA = False

    def __init__(self, dataset_model, gp, **kwargs):

        super().__init__(dataset_model, gp)

        # delta probability for calculation of beta
        self.delta_beta = kwargs['delta_beta'] if 'delta_beta' in kwargs.keys() else self.DELTA_BETA
        # self.is_constant_beta = kwargs['is_constant_beta'] if 'is_constant_beta' in kwargs.keys() \
        #     else self.DEFAULT_IS_CONSTANT_BETA
        #
        # if self.is_constant_beta:
        #     self.beta = kwargs['constant_beta_value'] if 'constant_beta_value' in kwargs.keys() \
        #                                                  and kwargs['constant_beta_value']\
        #         else self.DEFAULT_CONSTANT_BETA
        #
        # if self.is_constant_beta:
        #     is_default = 'default' if self.beta == self.DEFAULT_CONSTANT_BETA else ''
        #     log.info("Using {} constant beta={}".format(is_default, self.beta))
        # else:
        #     log.info("Using non-constant beta from original GP-UCB algorithm")
        log.info("Using non-constant beta from original GP-UCB algorithm")

    def predict(self, history, iteration):

        beta = self._get_beta(iteration)

        max_acc_value = -float('inf')
        best_location = None
        # to speed up
        cholesky = self.gp.cholesky(locations=history.locations)
        observed_locations = history.locations_set
        for i in range(self.dataset_model.num_points):
            current_loc = self.dataset_model.locations[i:i + 1, :]

            if tuple(current_loc[0, :]) in observed_locations:
                continue

            mean, var = self.gp.predict(locations=history.locations,
                                        measurements=history.measurements,
                                        current_location=current_loc,
                                        cholesky=cholesky)

            current_value = mean + beta * var

            if current_value > max_acc_value:
                max_acc_value = current_value
                best_location = current_loc

        return best_location

    def _get_beta(self, iteration):
        # if self.is_constant_beta:
        #     return self.beta
        # else:
            # todo note in this case we return sqrt beta, which is in the GP-UCB algorithm
            domain_size = self.dataset_model.num_points
            # iteration starts from zero
            squared_iteration = (iteration + 1) ** 2
            return sqrt(2 * math.log(domain_size * squared_iteration * (pi ** 2) / (6 * self.delta_beta)))
