from math import exp
import logging

from src.run.MultipleTestRunner import MultipleTestRunner

log = logging.getLogger(__name__)


def run_script(config, method):

    # create_logger(tests_root_folder=tests_root_folder,
    #               run_mode=method.method_type,
    #               eps_log=method.epsilon_log,
    #               r=method.r)

    log.info("Starting {} \n".format(method.method_string))

    epsilon = exp(method.epsilon_log)

    multiple_test_runner = MultipleTestRunner(dataset_type=config.DATASET_TYPE,
                                              root_folder=config.RESULTS_SAVE_ROOT_FOLDER,
                                              r=method.r,
                                              dataset_max_norm=config.DATASET_MAX_NORM)

    multiple_test_runner.run_test_for_all_seeds(seeds=config.SEEDS,
                                                r=method.r,
                                                epsilon=epsilon,
                                                delta=config.DELTA,
                                                num_iterations=config.NUM_ITERATIONS,
                                                dataset_filename=config.DATASET_FILENAME,
                                                initial_history_mode=config.INITIAL_HISTORY_MODE,
                                                run_mode=method.method_type)

    log.info("    <------------ End method ------------> \n\n")
