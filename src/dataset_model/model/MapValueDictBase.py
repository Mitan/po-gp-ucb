import numpy as np
from random import choice


class MapValueDictBase:
    # needed for rounding while adding into dict
    ROUNDING_CONST = 6

    def __init__(self, locations, values):
        self.num_points = locations.shape[0]
        self._locations = locations
        # the original mean of the values
        self.values_mean = np.mean(values)

        self.values = values
        self._vals_dict = {}

        self.update_val_dict()

    def update_val_dict(self):
        self._vals_dict = {}
        for i in range(self.num_points):
            rounded_location = np.around(self.locations[i], decimals=self.ROUNDING_CONST)
            self._vals_dict[tuple(rounded_location)] = self.values[i]

    @property
    def locations(self):
        return self._locations

    @locations.setter
    def locations(self, new_locations):
        # numbers of points should match
        assert self._locations.shape[0] == new_locations.shape[0],\
            "old locations shape {}, new locations shape {}".format(self.locations.shape, new_locations.shape)
        self._locations = new_locations
        self.update_val_dict()

    def get_random_start_location(self):
        return choice(list(self.locations))

    def __call__(self, query_location):
        assert query_location.ndim == 2
        assert query_location.shape[0] == 1
        query_location = query_location[0]
        tuple_loc = tuple(map(lambda x: round(x, ndigits=self.ROUNDING_CONST), query_location))

        assert tuple_loc in self._vals_dict, "No close enough match found for query location " + str(tuple_loc)
        return self._vals_dict[tuple_loc]

    def write_to_file(self, filename):
        vals = np.atleast_2d(self.values).T
        concatenated_dataset = np.concatenate((self.locations, vals), axis=1)
        np.savetxt(filename, concatenated_dataset, fmt='%11.8f', delimiter=',')

    def get_max(self):
        return max(self.values)