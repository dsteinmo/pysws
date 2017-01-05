import numpy as np
from numpy.fft import fft, ifft, fftshift, fft2, ifft2
from scipy.linalg import toeplitz

class FourierChebyshevSpatialDiscretization:
    def __init__(self, config):
        self.length_x = config['length_x']
        self.length_y = config['length_y']
        self.num_points_x = config['num_points_x']
        self.num_points_y = config['num_points_y']

        # self.__build_grid__()
        # self.__build_wavenumbers__()
        # self.__build_filter__()

    def cheb_dif(self, N, M):
        I = np.eye(N)

        n1 = np.floor(N/2)
        n2 = np.ceil(N/2)

        k = np.array([np.arange(0, N)]).T
        th = k*np.pi/(N-1)

        # Compute Chebyshev points.
        vec = np.arange(N-1, 1-N-1, -2)
        x = np.sin(np.pi*vec/(2*(N-1)))

        T = np.tile(th/2, (1, N))  # Like repmat(th/2, 1, N) for 2nd order tensors.
        Tt = T.T

        DX = 2*np.sin(Tt+T)*np.sin(Tt-T)
        DX = np.vstack([DX[0:n1, :], -np.flipud(np.fliplr(DX[0:n2, :]))])

        DX[range(N),range(N)] = 1.0

        C = toeplitz((-1.0)**k)
        C[0,:] = C[0,:]*2.0
        C[N-1,:] = C[N-1,:]*2.0
        C[:,0] = C[:,0] / 2.0
        C[:,N-1] = C[:,N-1] / 2.0

        Z = 1.0 / DX
        Z[range(N),range(N)] = 0.0

        D = np.eye(N)
        DM = np.zeros([N, N, M])

        for ell in range(1,M+1):
            D = ell*Z*(C*np.tile(np.array([np.diag(D)]).T,(1,N)) - D)
            diag = -np.sum(D,1)
            D[range(N),range(N)] = diag

            DM[:,:,ell-1] = D

        return (x,DM)