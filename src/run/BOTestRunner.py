from src.run.BORunner import BORunner
from src.utils.FileUtil import check_create_folder


class BOTestRunner:
    def __init__(self, bo_method):
        self.bo_method = bo_method

    def run_bo(self, seed_folder, model, gp, num_iterations,
               initial_history, output_filename,
               **kwargs):
        output_folder = seed_folder + 'result/'
        check_create_folder(output_folder)

        bo_runner = BORunner(method=self.bo_method,
                             dataset_model=model,
                             gp=gp,
                             **kwargs)
        history = bo_runner.run(num_iterations=num_iterations,
                                initial_history=initial_history
                                )

        history.write_to_file(max_dataset_value=model.get_max(),
                              filename=output_folder + output_filename)

        return history
