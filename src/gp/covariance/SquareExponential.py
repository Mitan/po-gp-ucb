import numpy as np


# non ARD kernel
class SquareExponential:
    def __init__(self, length_scale, signal_variance):
        self.ARD = False
        """
        @param: length_scale l - scalar
        @param: signal variance sigma_f_squared - float containing the signal variance
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        # const
        self.eps_tolerance = 10 ** -10

    def cov(self, physical_state_1, physical_state_2):
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        scaled_norm = np.linalg.norm(diff) / self.length_scale
        return self.signal_variance * np.exp(-0.5 * scaled_norm**2)
