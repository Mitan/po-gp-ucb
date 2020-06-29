from math import sqrt

import numpy as np

from src.enum.MethodEnum import MethodEnum
from src.utils.Util import transform_results_back


class ResultsExtractor:
    def __init__(self, dataset_type):
        # self.metric = metric
        self.dataset_type = dataset_type

    def _extract_results_for_one_method(self,
                                        method_name,
                                        results_filename,
                                        root_folder,
                                        seeds,
                                        num_iterations):
        len_seeds = len(seeds)
        all_regrets = np.zeros((len_seeds, num_iterations + 1))
        for ind, seed in enumerate(seeds):
            seed_folder = '{}seed{}/'.format(root_folder, seed)
            seed_regrets = self._extract_metric_for_one_method_seed(results_filename=results_filename,
                                                                    seed_folder=seed_folder)

            assert seed_regrets.shape[0] == num_iterations + 1
            all_regrets[ind, :] = seed_regrets
        mean = np.mean(all_regrets, axis=0)
        var = np.std(all_regrets, axis=0) / sqrt(len_seeds)
        return [method_name, mean, var]

    def extract_results(self, root_folder, seeds, num_iterations, method):
        if method.method_type == MethodEnum.UCB:
            results_filename = "ucb_history.txt"
        elif method.method_type == MethodEnum.ODP_GP_UCB:
            # method_name = r"$\epsilon = \exp({}), \ r = {}$".format(method.epsilon_log, method.r)
            results_filename = 'odp_gp_ucb_history_eps{:2.1f}_r{}.txt'.format(method.epsilon_log, method.r)
            # results_filename = 'ucb_history_eps{:2.1f}_r{}.txt'.format(method.epsilon_log, method.r)
        else:
            raise ValueError("Unknown method")

        return self._extract_results_for_one_method(method_name=method.plotting_method_string,
                                                    results_filename=results_filename,
                                                    root_folder=root_folder,
                                                    seeds=seeds,
                                                    num_iterations=num_iterations)

    def _extract_metric_for_one_method_seed(self, results_filename, seed_folder):
        result_folder = seed_folder + 'result/'

        with open(result_folder + results_filename, 'r') as f:
            lines = f.readlines()
        assert lines[-4].strip() == "Model max value"
        max_value = transform_results_back(value=float(lines[-3].strip()), dataset_type=self.dataset_type)
        history = lines[-1].strip()[1: -1].replace(",", "").split()
        # todo note that we transform the values back for plotting

        # these are instant regrets on every iteration
        regrets = [max_value - transform_results_back(value=float(x), dataset_type=self.dataset_type) for x in history]

        for j in range(1, len(regrets)):
            regrets[j] = min(regrets[j], regrets[j - 1])

        # if self.metric == PlottingMethods.SimpleRegret:
        #     for j in range(1, len(regrets)):
        #         regrets[j] = min(regrets[j], regrets[j - 1])
        # else:
        #     for j in range(1, len(regrets)):
        #         regrets[j] = (regrets[j] + j * regrets[j - 1]) / (j + 1)

        return np.array(regrets)
