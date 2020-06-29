import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class NormalMatrixGenerator:
    def __init__(self):
        pass
        # self.dataset_mode = dataset_mode

    def get_normals(self, folder, r, point_dimension):
        # NB change: instead of dataset mode, check that file exists
        normals_path = '{}normals_{}.txt'.format(folder, r)
        my_file = Path(normals_path)
        if my_file.is_file():
            normals = np.genfromtxt(fname=normals_path)
            log.info("Loading normals matrix")
        else:
            log.info("Generating normals matrix")
            normals = self._generate_normals_matrix(d=point_dimension, r=r,
                                                    filename=normals_path)
        """
        if self.dataset_mode == DatasetModeEnum.Load:
            normals = np.genfromtxt(fname=folder + 'normals.txt')
        else:

            normals = self._generate_normals_matrix(d=point_dimension, r=r,
                                                    filename=folder + 'normals.txt')
        """
        return normals

    @staticmethod
    def _generate_normals_matrix(d, r, filename):
        normals = np.random.normal(size=(d, r)) / math.sqrt(r)
        np.savetxt(filename, normals, fmt='%12.8f')
        return normals
