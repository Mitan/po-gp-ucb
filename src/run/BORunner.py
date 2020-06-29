import copy
from src.enum.AcquisitionEnum import AcquisitionEnum
from src.bo.methods.UCB import UcbBO
import logging
log = logging.getLogger(__name__)


class BORunner:
    def __init__(self, method, dataset_model, gp, **kwargs):
        self.model = dataset_model
        self.bo_method = self._get_bo_method(method, dataset_model, gp, **kwargs)

    @staticmethod
    def _get_bo_method(method, dataset_model, gp, **kwargs):
        if method == AcquisitionEnum.UCB:
            return UcbBO(dataset_model, gp, **kwargs)
        else:
            raise Exception("Unknown BO method")

    def run(self, num_iterations, initial_history):
        history = copy.deepcopy(initial_history)

        for i in range(num_iterations):
            new_location = self.bo_method.predict(history=history,
                                                  iteration=i)
            new_measurement = self.model(new_location)
            history.append(new_location=new_location,
                           new_measurement=new_measurement)

        log.debug("After running {} iterations, max found measurement is {:4f}, regret is {:4f}"
                          .format(num_iterations, max(history.measurements),
                                  self.model.get_max() - max(history.measurements)))
        return history
