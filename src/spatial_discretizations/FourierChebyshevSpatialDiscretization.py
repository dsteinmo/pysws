import numpy as np
from cheb.cheb import cheb_dif

class FourierChebyshevSpatialDiscretization:
    def __init__(self, config):
        self.length_x = config['length_x']
        self.length_y = config['length_y']
        self.num_points_x = config['num_points_x']
        self.num_points_y = config['num_points_y']

        self.__build_grid__()
        # self.__build_wavenumbers__()
        # self.__build_filter__()

    def __build_grid__(self):
        x_1d, Dm = cheb_dif(self.num_points_x, 1)

        y_1d, dy = np.linspace(0, self.length_y, self.num_points_y, False, True)
        self.x, self.y = np.meshgrid(x_1d, y_1d)


