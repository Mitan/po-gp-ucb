from src.dataset_model.DatasetGenerator import DatasetGenerator
from src.dataset_model.DatasetPreprocessor import DatasetPreprocessor
from src.dataset_model.DatasetTransformer import DatasetTransformer
from src.utils.FileUtil import check_create_folder

import math

import logging
log = logging.getLogger(__name__)


class ModelGenerator:
    def __init__(self, dataset_type, dataset_max_norm):
        self.dataset_type = dataset_type
        self.dataset_preprocessor = DatasetPreprocessor(dataset_max_norm)
        self.dataset_generator = DatasetGenerator(dataset_type=self.dataset_type,
                                                  dataset_max_norm=dataset_max_norm)

    @staticmethod
    def get_transformed_model(original_model, dataset_folder, normals, r, epsilon, delta):
        check_create_folder(dataset_folder)
        data_transformer = DatasetTransformer(r=r, epsilon=epsilon, delta=delta)
        transformed_model = data_transformer.transform_dataset(original_model, normals)
        transformed_model.write_to_file(filename='{}transformed_dataset_eps{:2.1f}_r{}.txt'.
                                        format(dataset_folder, math.log(epsilon), r))
        return transformed_model

    def get_original_model(self, **kwargs):

        original_model = self.dataset_generator.get_dataset_model(**kwargs)
        self.dataset_preprocessor.preprocess_locations(original_model)
        return original_model

        # if self.dataset_type == DatasetEnum.Simulated:
        #     if self.dataset_mode == DatasetModeEnum.Generate:
        #         log.info("Generating original model")
        #         original_model = self.dataset_generator.get_dataset_model(dataset_mode=self.dataset_mode)
        #         # note this method changes the original_model
        #         self.dataset_preprocessor.preprocess_locations(original_model)
        #         original_model.write_to_file(filename=dataset_folder + 'original_dataset.txt')
        #     else:
        #         log.info("Loading original model")
        #         model_filename = dataset_folder + '/original_dataset.txt'
        #         original_model = self.dataset_generator.get_dataset_model(dataset_mode=self.dataset_mode,
        #                                                                   filename=model_filename)
        # elif self.dataset_type == DatasetEnum.HousePrice or self.dataset_type == DatasetEnum.Loan:
        #     original_model = self.dataset_generator.get_dataset_model(dataset_mode=self.dataset_mode,
        #                                                               **kwargs)
        #     self.dataset_preprocessor.preprocess_locations(original_model)
        # else:
        #     raise ValueError("Unknown dataset")
        # return original_model
