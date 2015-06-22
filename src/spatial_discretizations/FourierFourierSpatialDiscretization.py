from pprint import pprint
import numpy as np
from numpy.fft import fft, ifft, fftshift

class FourierFourierSpatialDiscretization:

    def __init__(self, config):
        print "Hello, world. I'm FourierFourier!"
        self.length_x = config['length_x']
        self.length_y = config['length_y']
        self.num_points_x = config['num_points_x']
        self.num_points_y = config['num_points_y']

        self.__build_grid__()
        self.__build_wavenumbers__()

    def __build_grid__(self):
        x_1d, dx = np.linspace(0, self.length_x, self.num_points_x, False, True)
        y_1d, dy = np.linspace(0, self.length_y, self.num_points_y, False, True)
        self.x, self.y = np.meshgrid(x_1d, y_1d)

    def __build_wavenumbers__(self):
        k_1d = fftshift(np.array(range(0, self.num_points_x))*
                                               ((2*np.pi)/self.length_x))
        l_1d = fftshift(np.array(range(0, self.num_points_y))*
                                               ((2*np.pi)/self.length_y))
        self.k, self.l = np.meshgrid(k_1d, l_1d)


