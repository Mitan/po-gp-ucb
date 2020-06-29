from src.enum.DatasetEnum import DatasetEnum
from src.enum.InitialHistoryModeEnum import InitialHistoryModeEnum
from src.enum.MethodEnum import MethodEnum
from src.run.MethodDescriptor import MethodDescriptor


class Config:

    def __init__(self):
        pass

    # path
    RESULTS_SAVE_ROOT_FOLDER = './tests/house/'

    DATASET_TYPE = DatasetEnum.HousePrice

    SEEDS = range(50)
    NUM_ITERATIONS = 100

    DELTA = 10 ** (-4)

    DATASET_MAX_NORM = 25.0

    DATASET_FILENAME = './datasets/house/house_dataset_x25.csv'

    INITIAL_HISTORY_MODE = InitialHistoryModeEnum.Random

    METHODS = [MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=-1.0,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=0.0,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=0.5,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=1.0,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=2.0,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=2.8,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.UCB)
               ]
