import logging
from pathlib import Path
from random import randint

import numpy as np

from src.enum.InitialHistoryModeEnum import InitialHistoryModeEnum
from src.utils.History import History

log = logging.getLogger(__name__)


class InitialHistoryGenerator:
    def __init__(self):
        pass

    @staticmethod
    def _get_initial_history_index(num_points, initial_history_mode, folder):
        filename = '{}initial_history_index.txt'.format(folder)

        my_file = Path(filename)
        if my_file.is_file():
            with open(filename, 'r') as f:
                index = int(f.readline())
            log.info("Loading initial history with index {}".format(index))
        else:
            # middle point index
            if initial_history_mode == InitialHistoryModeEnum.Deterministic:
                index = num_points // 2
                log.info("Generating initial deterministic history with index {}".format(index))
            elif initial_history_mode == InitialHistoryModeEnum.Random:
                index = randint(0, num_points - 1)
                log.info("Generating initial random history with index {}".format(index))
            else:
                raise Exception("Wrong initial history mode")

            with open(filename, 'w') as f:
                f.write(str(index))

        return index

    @staticmethod
    def get_history(model, index):
        initial_point = model.locations[index: index + 1, :]
        assert initial_point.ndim == 2
        initial_measurement = model(initial_point)
        return History(initial_locations=initial_point,
                       initial_measurements=np.array([initial_measurement]))

    def generate_initial_history(self, model, initial_history_mode, folder):
        num_points = model.num_points
        index = self._get_initial_history_index(num_points, initial_history_mode, folder)
        return self.get_history(model, index)
