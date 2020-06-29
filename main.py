from config import Config
from src.plotting.plotting_script import plot_results
from src.run.run_script import run_script


from src.utils.FileUtil import create_logger, check_create_folder

if __name__ == '__main__':

    config = Config()

    check_create_folder(folder_name=config.RESULTS_SAVE_ROOT_FOLDER)

    create_logger(tests_root_folder=config.RESULTS_SAVE_ROOT_FOLDER)

    for method in config.METHODS:
        run_script(config=config,
                   method=method)

    plot_results(config)
