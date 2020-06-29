import numpy as np


class History:
    ROUNDING_CONST = 6

    def __init__(self, initial_locations, initial_measurements):
        self.locations = initial_locations
        self.measurements = initial_measurements

        assert self.locations.ndim == 2
        assert self.measurements.ndim == 1
        assert self.locations.shape[0] == self.measurements.shape[0]

        self.dim = self.locations.shape[1]

        locations_set = [tuple(initial_locations[i,:]) for i in range(initial_locations.shape[0])]
        self.locations_set = set(locations_set)

    def append(self, new_location, new_measurement):
        """
        new_measurements - 1D array
        new_locations - 2D array
        @modifies - self.locations, self.measurements
        """
        assert new_location.ndim == 2
        assert self.locations.ndim == 2
        assert new_location.shape[1] == self.dim

        self.locations = np.append(self.locations, new_location, axis=0)
        # 1D array

        assert self.measurements.ndim == 1

        self.measurements = np.append(self.measurements, new_measurement)
        self.locations_set.add(tuple(new_location[0, :]))

    def write_to_file(self, filename, max_dataset_value):
        num_points, d = self.locations.shape

        with open(filename, 'w') as f:
            f.write('number of points={}\n'.format(num_points))
            f.write('point dimension={}\n'.format(d))
            for i in range(num_points):
                f.write(" {} {:6f} \n".
                        format(" ".join(str(round(x, self.ROUNDING_CONST)) for x in list(self.locations[i,:])),
                               self.measurements[i]))

            f.write("Locations selected \n")
            list_locations = [list(map(lambda t :round(t, self.ROUNDING_CONST), x)) for x in self.locations.tolist()]
            for i in range(num_points):
                f.write("{} \n".format(list_locations[i]))

            f.write("Model max value\n")
            f.write("{:6f} \n".format(max_dataset_value))

            f.write("Measurements found \n")
            f.write('{} \n'.format([round(x, self.ROUNDING_CONST) for x in self.measurements.tolist()]))



