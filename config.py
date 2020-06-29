from src.enum.DatasetEnum import DatasetEnum
from src.enum.InitialHistoryModeEnum import InitialHistoryModeEnum
from src.enum.MethodEnum import MethodEnum
from src.run.MethodDescriptor import MethodDescriptor


class Config:

    def __init__(self):
        pass

    # path
    RESULTS_SAVE_ROOT_FOLDER = './tests/toy_example/'

    DATASET_TYPE = DatasetEnum.Loan

    SEEDS = range(20)
    NUM_ITERATIONS = 40

    DELTA = 0.01

    # normalize the norms of the dataset matrix to make them consistent with DP definition
    DATASET_MAX_NORM = 25

    DATASET_FILENAME = './datasets/toy_example/test.csv'

    INITIAL_HISTORY_MODE = InitialHistoryModeEnum.Random

    METHODS = [MethodDescriptor(method_type=MethodEnum.UCB),

               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=3.8,
                                r=15),
               MethodDescriptor(method_type=MethodEnum.ODP_GP_UCB,
                                epsilon_log=2.0,
                                r=15)
               ]
