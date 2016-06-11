#!/usr/bin/python2.7
'''
Written by Derek Steinmoeller, 2015.
'''

import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt

# Run-specific constants.
LX = 3.0e3
LY = 3.0e3
G = 9.81
F = 0.0
FINAL_TIME = 300.0
NX = 128
NY = 128
CFL = 0.45 
CONST_DEPTH = 10.0
LOGGING_INTERVAL = 10
ONLINE_VISUALIZATION = True

# Some module config.
np.set_printoptions(threshold='nan', precision=4)
plt.ioff()

# Build physical grid and grid in spectral space.
x_1d, dx = np.linspace(0, LX, NX, False, True)
y_1d, dy = np.linspace(0, LY, NY, False, True)
x, y = np.meshgrid(x_1d, y_1d)

k_1d = fftshift(np.array(range(0 ,NX))*((2*np.pi)/LX))
l_1d = fftshift(np.array(range(0, NY))*((2*np.pi)/LY))
k, l = np.meshgrid(k_1d, l_1d)

# Declare background depth profile.
background_depth = CONST_DEPTH - 3.0*np.exp(-np.square((x - 0.5*LX)/(LX/5.0)))

# Compute bed slopes.
background_depth_xderiv = np.real(ifft(1.j*k*fft(background_depth, axis=1), axis=1))
background_depth_yderiv = np.real(ifft(1.j*l*fft(background_depth, axis=0), axis=0))

# Declare initial conditions.
eta_initial = 1.e-1*np.exp(-np.square((x - 0.5*LX)/(LX/15.0)) - 
                     np.square((y - 0.5*LY)/(LY/15.0)))
u_initial = np.zeros([NY, NX])
v_initial = np.zeros([NY, NX])

# Initialize all storage needed for SW equations.
NUM_FIELDS = 3
q = np.zeros([NY, NX, NUM_FIELDS])
rhs_q = np.zeros([NY, NX, NUM_FIELDS])
res_q = np.zeros([NY, NX, NUM_FIELDS])
flux_qx = np.zeros([NY, NX, NUM_FIELDS])
flux_qy = np.zeros([NY, NX, NUM_FIELDS])
div_q = np.zeros([NY, NX, NUM_FIELDS])
source = np.zeros([NY, NX, NUM_FIELDS])

# Set initial conditions for SW.
q[:,:,0] = background_depth + eta_initial  # h
q[:,:,1] = q[:,:,0]*u_initial              # hu 
q[:,:,2] = q[:,:,0]*v_initial              # hv

# Define time-stepper: LSERK4.
LSERK4_STAGES = range(0,5)
RK4A = [            0.0,
        -567301805773.0/1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/842570457699.0];
        
RK4B = [ 1432997174477.0/9575080441755.0,
         5161836677717.0/13612068292357.0,
         1720146321549.0/2090206949498.0,
         3134564353537.0/4481467310338.0,
         2277821191437.0/14882151754819.0];


time = 0.0
count = 0
while time < FINAL_TIME:
    
    # calculate shallow water time step.
    max_wave_speed = np.sqrt(G*np.max(q[:,:,0]))
    dt = (np.min(np.array([dx, dy])) / max_wave_speed)*CFL
        
    # Standard logging output
    if ((count % LOGGING_INTERVAL) == 0)  or count == 1:
        print("t: ", time, ", dt: ", dt, " max h: ", np.max(q[:,:,0]), " max hu: ", np.max(q[:,:,1]))
        
        if ONLINE_VISUALIZATION == True:
            plt.clf()
            plt.pcolor(x, y, q[:,:,0] - background_depth)
            plt.colorbar()
            plt.draw()
            plt.ion()
            plt.show()
            plt.pause(0.001)
    
    # Loop through Runge-Kutta stages.    
    for intrk in LSERK4_STAGES:
        # compute right hand side of sw equations
        # Compute shallow water flux vectors and source term.
        # [ hu                      [ hv
        #   hu^2 + 0.5*g*h^2     &    huv
        #   huv              ]        hv^2 + 0.5*g*h^2 ]
        flux_qx[:,:,0] = q[:,:,1]                               
        flux_qx[:,:,1] = q[:,:,1]*q[:,:,1]/q[:,:,0] + 0.5*G*q[:,:,0]*q[:,:,0]
        flux_qx[:,:,2] = q[:,:,1]*q[:,:,2]/q[:,:,0] 
        
        flux_qy[:,:,0] = q[:,:,2]                        
        flux_qy[:,:,1] = q[:,:,1]*q[:,:,2]/q[:,:,0]
        flux_qy[:,:,2] = q[:,:,2]*q[:,:,2]/q[:,:,0] + 0.5*G*q[:,:,0]*q[:,:,0] 
           
        source[:,:,0] = np.zeros([NY, NX])
        source[:,:,1] = G*q[:,:,0]*background_depth_xderiv - F*q[:,:,2]
        source[:,:,2] = G*q[:,:,0]*background_depth_yderiv + F*q[:,:,1]
                       
        for i in range(0, NUM_FIELDS):
            # Compute divergence of flux (axis=1 is x, axis=0 is y).
            div_q[:,:,i] = np.real(ifft((1.j*k)*fft(flux_qx[:,:,i], axis=1), axis=1)) +  np.real(ifft((1.j*l)*fft(flux_qy[:,:,i], axis=0), axis=0))  
        
        # Compute RHS of PDE (LHS contains only dq/dt).    
        rhs_q = -div_q + source

        # Compute Runge-Kutta residual.
        res_q = RK4A[intrk]*res_q + dt*rhs_q

        # Update fields.
        q += RK4B[intrk]*res_q
    
    # prepare for next time-step.
    time += dt
    count += 1

print ("======================================================="
       "=======================")
print("Simulation complete.")
plt.show()
