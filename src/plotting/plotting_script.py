from src.plotting.ResultsPlotter import ResultGraphPlotter
from src.run.ResultsExtractor import ResultsExtractor
from src.utils.FileUtil import generate_file_handle


def plot_results(config):
    result_extractor = ResultsExtractor(dataset_type=config.DATASET_TYPE)

    all_results = []
    for method in config.METHODS:
        all_results.append(result_extractor.extract_results(root_folder=config.RESULTS_SAVE_ROOT_FOLDER,
                                                            seeds=config.SEEDS,
                                                            num_iterations=config.NUM_ITERATIONS,
                                                            method=method))

    for result in all_results:
        mean = result[1][-1]
        print("Simple regret of {} is {:10.4f}".
              format(result[0], mean))
    print()

    results_plotter = ResultGraphPlotter(dataset_type=config.DATASET_TYPE,
                                         num_iterations=config.NUM_ITERATIONS)

    results_filename = generate_file_handle(dataset_type=config.DATASET_TYPE)

    results_plotter.plot_results(results=all_results,
                                 plot_bars=True,
                                 output_file_name=config.RESULTS_SAVE_ROOT_FOLDER + results_filename)
