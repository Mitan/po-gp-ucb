import copy
import math

import numpy as np
import logging
log = logging.getLogger(__name__)


class DatasetTransformer:
    def __init__(self, r, epsilon, delta):
        self.r = r
        self.epsilon = epsilon
        self.delta = delta
        # threshold
        self.w = 16 * math.sqrt(self.r * math.log(2 / self.delta)) \
                 * math.log(16 * self.r / self.delta) \
                 / self.epsilon

        log.info("log_epsilon = {}, delta = {}".format(math.log(epsilon), delta))
        log.info("Threshold omega = {}".format(self.w))

    def transform_dataset(self, model, normals):
        points = model.locations
        n, original_d = points.shape
        # assert np.mean(points) == 0

        u, s, v = np.linalg.svd(points, full_matrices=True)
        # print(u.shape, s.shape, v.shape)
        # shift the values if the are not large enough
        log.info("min singular value is {}".format(min(s)))
        if min(s) < self.w:
            log.info("Shifting the singular values")
            s += self.w
            smat = np.zeros((n, original_d), dtype=float)
            min_n_d = min(n, original_d)
            smat[:min_n_d, :min_n_d] = np.diag(s)
            points = np.dot(u, np.dot(smat, v))
        else:
            log.info("Not shifting the singular values")
        """
        if normals is None:
            assert normals_filename is not None
            normals = self.generate_normals_matrix(original_d, self.r, normals_filename)
        """
        transformed = np.dot(points, normals)
        new_model = copy.deepcopy(model)
        new_model.locations = transformed
        return new_model

