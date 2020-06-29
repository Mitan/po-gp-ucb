import logging
import os

from src.enum.DatasetEnum import DatasetEnum
from src.enum.MethodEnum import MethodEnum

log = logging.getLogger(__name__)


def write_results_to_file(means, variances, filename):
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'

    with open(filename, append_write) as f:
        # f.write(method + '\n')
        f.write("Means \n")
        f.write('\n'.join(map(str, list(means))) + '\n')
        f.write("Variances \n")
        f.write('\n'.join(map(str, list(variances))) + '\n')


def check_create_folder(folder_name):
    # print("seed is {}".format(seed))
    try:
        os.makedirs(folder_name)
    except OSError:
        if not os.path.isdir(folder_name):
            raise


def create_logger(tests_root_folder):

    logging_folder = "{}logs/".format(tests_root_folder)
    check_create_folder(folder_name=logging_folder)
    logging_filename = '{}logging.conf'.format(logging_folder)
    logging.basicConfig(filename=logging_filename, level=logging.INFO)


def create_tests_root_folder(save_results_root_folder, subfolder_name):
    check_create_folder(folder_name=save_results_root_folder)

    tests_root_folder = '{}{}/'.format(save_results_root_folder, subfolder_name)
    # tests_root_folder = Config.RESULTS_SAVE_ROOT_FOLDER

    check_create_folder(folder_name=tests_root_folder)
    return tests_root_folder


def generate_file_handle(dataset_type):
    metric_string = "simple_regrets"
    if dataset_type == DatasetEnum.HousePrice:
        dataset_string = "house"
    elif dataset_type == DatasetEnum.Simulated:
        dataset_string = "simulated"
    elif dataset_type == DatasetEnum.Loan:
        dataset_string = "loan"
    elif dataset_type == DatasetEnum.Branin:
        dataset_string = "branin"
    else:
        raise ValueError("Unknown dataset")
    return "{}_{}.eps".format(dataset_string, metric_string)
