import logging

import numpy as np

from src.dataset_model.HypersStorerFactory import get_hyper_storer
from src.enum.AcquisitionEnum import AcquisitionEnum
from src.enum.MethodEnum import MethodEnum
from src.gp.GaussianProcess import GaussianProcess
from src.run.InitialHistoryGenerator import InitialHistoryGenerator
from src.run.ModelGenerator import ModelGenerator
from src.run.NormalMatrixGenerator import NormalMatrixGenerator
from src.run.BOTestRunner import BOTestRunner
from src.utils.FileUtil import check_create_folder
from src.utils.Util import process_results, get_results_filenames

log = logging.getLogger(__name__)


class MultipleTestRunner:
    def __init__(self, dataset_type, root_folder, r, dataset_max_norm):

        self.r = r
        self.root_folder = root_folder
        check_create_folder(self.root_folder)
        self.dataset_type = dataset_type

        self.hyper_storer = get_hyper_storer(dataset_type=self.dataset_type,
                                             dataset_max_norm=dataset_max_norm)
        self.gp = GaussianProcess(hyper_storer=self.hyper_storer,
                                  mean_function=self.hyper_storer.mean_function,
                                  noise_variance=self.hyper_storer.noise_variance)

        self.model_generator = ModelGenerator(dataset_type=self.dataset_type,
                                              dataset_max_norm=dataset_max_norm)
        self.history_generator = InitialHistoryGenerator()
        self.normals_generator = NormalMatrixGenerator()
        self.bo_runner = BOTestRunner(bo_method=AcquisitionEnum.UCB)

    def run_test_for_all_seeds(self,
                               seeds,
                               num_iterations,
                               epsilon, delta,
                               run_mode,
                               initial_history_mode,
                               **kwargs):

        len_seeds = len(seeds)
        all_regrets = np.zeros((len_seeds, num_iterations + 1))

        results_folder = "{}results/".format(self.root_folder)
        check_create_folder(results_folder)

        seed_result_file, results_filename = get_results_filenames(run_mode, results_folder, epsilon, self.r)

        for ind, seed in enumerate(seeds):
            log.info("Starting processing seed {}".format(seed))

            seed_folder = '{}seed{}/'.format(self.root_folder, seed)
            check_create_folder(seed_folder)

            model = self.model_generator.get_original_model(
                dataset_save_folder=seed_folder + 'datasets/',
                **kwargs)

            if run_mode == MethodEnum.ODP_GP_UCB:

                normals = self.normals_generator.get_normals(folder=seed_folder,
                                                             r=self.r,
                                                             point_dimension=model.point_dimension)

                model = self.model_generator.get_transformed_model(original_model=model,
                                                                   dataset_folder=seed_folder + 'datasets/',
                                                                   normals=normals,
                                                                   epsilon=epsilon,
                                                                   delta=delta,
                                                                   r=self.r)

            all_regrets[ind, :] = self._run_test_for_one_seed(result_filename=seed_result_file,
                                                              model=model,
                                                              seed_folder=seed_folder,
                                                              initial_history_mode=initial_history_mode,
                                                              num_iterations=num_iterations,
                                                              **kwargs)
            log.info("Finished processing seed {} \n".format(seed))

        process_results(regrets=all_regrets,
                        results_filename=results_filename)

    def _run_test_for_one_seed(self,
                               result_filename,
                               model,
                               seed_folder,
                               initial_history_mode,
                               num_iterations,
                               **kwargs):
        # original_output_filename = 'original_history_eps{:2.1f}.txt'.format(math.log(epsilon))
        original_initial_history = \
            self.history_generator.generate_initial_history(model=model,
                                                            initial_history_mode=initial_history_mode,
                                                            folder=seed_folder)
        original_history = self.bo_runner.run_bo(seed_folder=seed_folder,
                                                 gp=self.gp,
                                                 initial_history=original_initial_history,
                                                 model=model,
                                                 num_iterations=num_iterations,
                                                 output_filename=result_filename,
                                                 **kwargs)
        return model.get_max() - original_history.measurements
