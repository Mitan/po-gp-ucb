import logging
from pathlib import Path

from src.dataset_model.model.RealWorldDatasetMapValueDict import RealWorldDatasetMapValueDict
from src.enum.DatasetEnum import DatasetEnum
from src.dataset_model.model.SimulatedDomainDescriptor import SimulatedDomainDescriptor
from src.dataset_model.HypersStorerFactory import get_hyper_storer
from src.dataset_model.model.SimulatedMapValueDict import SimulatedMapValueDict
from src.utils.FileUtil import check_create_folder

log = logging.getLogger(__name__)


class DatasetGenerator:

    def __init__(self, dataset_type, dataset_max_norm):
        self.dataset_max_norm = dataset_max_norm
        log.info("Dataset max norm is {}".format(dataset_max_norm))
        self.type = dataset_type
        self.hyper_storer = get_hyper_storer(dataset_type, dataset_max_norm=dataset_max_norm)
        self.hyper_storer.print_params()

    def get_dataset_model(self, **kwargs):
        if self.type == DatasetEnum.Simulated:
            return self._get_simulated_dataset_model(**kwargs)
        elif self.type == DatasetEnum.HousePrice or self.type == DatasetEnum.Loan or DatasetEnum.Branin:
            return self._get_real_world_dataset_model(**kwargs)
        else:
            raise ValueError("Unknown dataset")

    def _get_real_world_dataset_model(self, **kwargs):
        m = RealWorldDatasetMapValueDict(hyper_storer=self.hyper_storer,
                                         filename=kwargs['dataset_filename'])
        return m

    def _get_simulated_dataset_model(self, **kwargs):

        domain_descriptor = SimulatedDomainDescriptor(dataset_max_norm=self.dataset_max_norm)
        assert self.hyper_storer.ARD is False

        dataset_folder = kwargs['dataset_save_folder']
        check_create_folder(dataset_folder)
        model_filename = dataset_folder + '/original_dataset.txt'

        my_file = Path(model_filename)
        if my_file.is_file():
            log.info("Loading original model")
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor,
                                      filename=model_filename)
        else:
            log.info("Generating original model")
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor)
            m.write_to_file(filename=model_filename)

        return m
