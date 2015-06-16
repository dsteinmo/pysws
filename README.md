## Synopsis

pysw is a simple python script for solving the shallow water equations in a doubly-periodic geometry. The method is based on the Fourier pseudospectral method (see `Spectral Methods in Matlab` by L.N. Trefethen).

## Code Example

pysw illustrates how simple it is to do Matlab-type computations with numpy. Here is an example from the code where a gradient is calculated with Discrete Fourier transforms.

<pre>
# Declare background depth profile.
background_depth = CONST_DEPTH - 3.0*np.exp(-np.square((x - 0.5*LX)/(LX/5.0)))

# Compute bed slopes.
background_depth_xderiv = np.real(ifft(1.j*k*fft(background_depth, axis=1), axis=1))
background_depth_yderiv = np.real(ifft(1.j*l*fft(background_depth, axis=0), axis=0))
</pre>

## Motivation

This project exists a spin-off from my Ph.D. thesis, that involved a lot of work with Shallow Water-type equations in two dimenions.

## Installation and Running

1. Install dependencies (python2.7, numpy, matplotlib).
2. Download the source.
3. Run with `python pysw.py`.
4. Start playing with the source code!

## Tests

The code's current architecture doesn't support automated unit testing.

## Contributors

Derek Steinmoeller

## License

Licensed under the MIT license.
