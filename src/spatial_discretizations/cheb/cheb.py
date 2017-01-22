import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import fft


def cheb_dif(N, M):
    I = np.eye(N)

    n1 = int(np.floor(N / 2))
    n2 = int(np.ceil(N / 2))

    k = np.array([np.arange(0, N)]).T
    th = k * np.pi / (N - 1)

    # Compute Chebyshev points.
    vec = np.arange(N - 1, 1 - N - 1, -2)
    x = np.sin(np.pi * vec / (2 * (N - 1)))

    T = np.tile(th / 2, (1, N))  # Like repmat(th/2, 1, N) for 2nd order tensors.
    Tt = T.T

    DX = 2 * np.sin(Tt + T) * np.sin(Tt - T)
    DX = np.vstack([DX[0:n1, :], -np.flipud(np.fliplr(DX[0:n2, :]))])

    DX[range(N), range(N)] = 1.0

    C = toeplitz((-1.0) ** k)
    C[0, :] = C[0, :] * 2.0
    C[N - 1, :] = C[N - 1, :] * 2.0
    C[:, 0] = C[:, 0] / 2.0
    C[:, N - 1] = C[:, N - 1] / 2.0

    Z = 1.0 / DX
    Z[range(N), range(N)] = 0.0

    D = np.eye(N)
    DM = np.zeros([N, N, M])

    for ell in range(1, M + 1):
        D = ell * Z * (C * np.tile(np.array([np.diag(D)]).T, (1, N)) - D)
        diag = -np.sum(D, 1)
        D[range(N), range(N)] = diag

        DM[:, :, ell - 1] = D

    return (x, DM)

def cosine_transform(f, axis=0):
    """
    :type f: numpy.array, 2D.
    """
    if axis == 1:
        f = f.T

    sz = f.shape
    N = sz[0]
    even_extension = np.vstack((f, np.flipud(f[1:N - 1, :])))
    cc = fft(even_extension, axis=0)
    result = cc[:N, :]
    if np.isreal(f).all():
        result = np.real(result)

    if axis == 1:
        return result.T

    return result

def cheb_derivative(f, axis=0):
    if axis == 1:
        f = f.T

    sz = f.shape
    N = sz[0]

    cc = cosine_transform(np.real(f))
    dd = np.zeros(f.shape)

    dd[N-2, :] = 2*(N-1)*cc[N-1, :]
    for k in range(N-3, -1, -1):
        dd[k, :] = dd[k+2, :] + 2*(k+1)*cc[k+1, :]

    dd = cosine_transform(np.real(dd))/(2*(N-1))

    if axis == 1:
        return dd.T

    return dd

