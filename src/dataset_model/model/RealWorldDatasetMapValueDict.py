import numpy as np

from src.dataset_model.model.MapValueDictBase import MapValueDictBase
from src.enum.DatasetEnum import DatasetEnum


class RealWorldDatasetMapValueDict(MapValueDictBase):

    def __init__(self, hyper_storer, filename):
        self.dataset_type = DatasetEnum.HousePrice
        self.hyper_storer = hyper_storer

        data = np.genfromtxt(filename, delimiter=',')
        locs = data[:, :-1]
        vals = data[:, -1]

        self.point_dimension = locs.shape[1]

        MapValueDictBase.__init__(self, locations=locs, values=vals)