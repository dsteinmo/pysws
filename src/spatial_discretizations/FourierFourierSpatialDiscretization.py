import numpy as np
from numpy.fft import fft, ifft, fftshift, fft2, ifft2

class FourierFourierSpatialDiscretization:

    def __init__(self, config):
        self.length_x = config['length_x']
        self.length_y = config['length_y']
        self.num_points_x = config['num_points_x']
        self.num_points_y = config['num_points_y']

        self.__build_grid__()
        self.__build_wavenumbers__()
        self.__build_filter__()

    def __build_grid__(self):
        x_1d, dx = np.linspace(0, self.length_x, self.num_points_x, False, True)
        y_1d, dy = np.linspace(0, self.length_y, self.num_points_y, False, True)
        self.x, self.y = np.meshgrid(x_1d, y_1d)

        self.min_grid_spacing = min(dx,dy)

    def __build_wavenumbers__(self):
        k_1d = fftshift(np.array(range(0, self.num_points_x))*
                                               ((2*np.pi)/self.length_x))
        l_1d = fftshift(np.array(range(0, self.num_points_y))*
                                               ((2*np.pi)/self.length_y))

        self.kmax = max(k_1d)
        self.lmax = max(l_1d)
        self.k, self.l = np.meshgrid(k_1d, l_1d)


    def __build_filter__(self):
        cutoff = 0.65 # 0.65 typically,  CUTOFF OF 0 CORRESPONDS TO HYPERVISCOSITY
        epsf = 1e-15 # FILTER STRENGTH AT HIGH WAVENUMBERS

        kcrit = self.kmax*cutoff
        lcrit = self.lmax*cutoff
        filter_order = 4

        self.filter = np.ones([self.num_points_y, self.num_points_x])
        # Filter in x.
        mask = np.abs(self.k) < kcrit
        self.filter *= (mask + (1-mask)*np.exp(np.log(epsf)*(np.power((self.k-kcrit)/(self.kmax-kcrit), filter_order))))

        # Filter in y.
        mask = np.abs(self.l) < lcrit
        self.filter *= (mask + (1-mask)*np.exp(np.log(epsf)*(np.power((self.l-lcrit)/(self.lmax-lcrit), filter_order))))

        # Remove the Nyquist frequency entirely
        self.filter[:, np.floor(self.num_points_x/2+1)] = 0
        self.filter[np.floor(self.num_points_y/2+1), :] = 0


    def differentiate_x(self, field):
        return np.real(ifft((1.j*self.k)*fft(field, axis=1), axis=1))

    def differentiate_y(self, field):
        return np.real(ifft((1.j*self.l)*fft(field, axis=0), axis=0))

    def filter_field(self, field):
        return np.real(ifft2(self.filter*fft2(field)))
