import numpy as np

import logging
log = logging.getLogger(__name__)


class DatasetPreprocessor:
    def __init__(self, dataset_max_norm):
        self.dataset_max_norm = dataset_max_norm

    # set mean of locations to zero and max norm to the given constant
    def preprocess_locations(self, model):
        if self.dataset_max_norm:
            log.info("Pre-processing locations to zero mean and {} max norm".format(self.dataset_max_norm))
            max_norm = max(np.linalg.norm(model.locations, axis=1))
            mean_locations = np.mean(model.locations, axis=0)

            model.locations = (model.locations - mean_locations) * (self.dataset_max_norm / max_norm)
        # tolerance_eps = 10 ** -6
        # if max_norm <= 1 + tolerance_eps and np.linalg.norm(mean_locations) < tolerance_eps:
        #     log.info("Not pre-processing locations")
        else:
            log.info("Not pre-processing locations")
