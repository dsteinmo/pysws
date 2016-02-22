#!/usr/bin/python2.7
'''
Written by Derek Steinmoeller, 2015/2016.
'''

import numpy as np
#from numpy.fft import fft, ifft, fftshift
#import matplotlib.pyplot as plt
from solvers.sw_solver import SW_Solver

import os

run_config_path = os.path.join(os.getcwd(), 'src/sample_run.json')


solver = SW_Solver(run_config_path)

# Define some short-forms.
x = solver.spatial_discretization.x
y = solver.spatial_discretization.y
num_x = solver.spatial_discretization.num_points_x
num_y = solver.spatial_discretization.num_points_y
LX = solver.spatial_discretization.length_x
LY = solver.spatial_discretization.length_y

# Define background depth
H0 = 10.0
background_depth = H0 - 3.0*np.exp(-np.square((x - 0.5*LX)/(LX/5.0)))

# Define initial conditions and initialize the solver.
eta_initial = 1.e-1*np.exp(-np.square((x - 0.5*LX)/(LX/15.0)) -
                     np.square((y - 0.5*LY)/(LY/15.0)))
u_initial = np.zeros([num_y, num_x])
v_initial = np.zeros([num_y, num_x])

solver.initialize_fields(eta_initial, u_initial, v_initial, background_depth)

solver.solve()
