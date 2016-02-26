import json
from pprint import pprint

import numpy as np


class SW_Solver:
    NUM_FIELDS = 3

    def __init__(self, config_file_path):
        print "config_file is:", config_file_path
        
        config_file = open(config_file_path, 'r')
        
        config = json.load(config_file)

        pprint(config)

        self.g = config['phys_params']['g']
        self.f = config['phys_params']['f']

        self.spatial_params = config['spatial_params']
        self.temporal_params = config['temporal_params']

        spatial_discretization_name = \
                  self.spatial_params['discretization_method']
        print "Loading spatial discretization: ", spatial_discretization_name
        
        SpatialDiscretization = self.__load_spatial_discretization__(
                                       spatial_discretization_name)

        time_stepper_name = self.temporal_params['time_stepper']
        print "Loading time-stepper: ", time_stepper_name

        TimeStepper = self.__load_time_stepper__(time_stepper_name)

        # Instantiate spatial discretization.
        self.spatial_discretization = SpatialDiscretization(
                                                    self.spatial_params)

        # Initiate time stepper.
        self.time_stepper = TimeStepper()

        # Initiate console logger.
        logger_name = config['console_logger']['name']
        ConsoleLogger = self.__load_outputter__(logger_name)
        self.console_logger = ConsoleLogger(config['console_logger'])

        self.__initialize_storage__()

        self.time=0.0
        self.final_time = self.temporal_params['final_time']
        self.cfl = self.temporal_params['cfl']

    def initialize_fields(self, eta_initial, u_initial, v_initial,
                           background_depth):

        self.q[:,:,0] = background_depth + eta_initial
        self.q[:,:,1] = self.q[:,:,0]*u_initial
        self.q[:,:,2] = self.q[:,:,0]*v_initial

        self.background_depth_xderiv = \
              self.spatial_discretization.differentiate_x(background_depth)
        
        self.background_depth_yderiv = \
              self.spatial_discretization.differentiate_y(background_depth)

    def solve(self):
        count = 0
        while self.time < self.final_time:

            self.compute_max_wave_speed()

            self.compute_time_step()

            outputmessage = '{0}, {1}, {2}'.format(count, self.time, self.final_time)
            self.console_logger.output(count, outputmessage)

            self.compute_rhs()

            self.time_stepper.step(self.q, self.rhs, self.temp, self.dt)

            count += 1
            self.time += self.dt

    def compute_max_wave_speed(self):
        self.max_wave_speed = np.sqrt(self.g*np.max(self.q[:,:,0]))

    def compute_time_step(self):
        self.dt = (self.spatial_discretization.min_grid_spacing 
                  / self.max_wave_speed)*self.cfl

    def compute_rhs(self):
        self.__compute_flux__()
        self.__compute_flux_divergence__()
        self.__compute_source__()

        self.rhs = -self.div_flux_q + self.source

    def __compute_flux__(self):
        q = self.q
        g = self.g
        self.flux_qx[:,:,0] = q[:,:,1]
        self.flux_qx[:,:,1] = q[:,:,1]*q[:,:,1]/q[:,:,0] \
                            + 0.5*g*q[:,:,0]*q[:,:,0]

        self.flux_qx[:,:,2] = q[:,:,1]*q[:,:,2]/q[:,:,0]

        self.flux_qy[:,:,0] = q[:,:,2]
        self.flux_qy[:,:,1] = q[:,:,1]*q[:,:,2]/q[:,:,0]
        self.flux_qy[:,:,2] = q[:,:,2]*q[:,:,2]/q[:,:,0] \
                            + 0.5*g*q[:,:,0]*q[:,:,0]


    def __compute_flux_divergence__(self):
        sd = self.spatial_discretization

        for i in range(0, SW_Solver.NUM_FIELDS):
            self.div_flux_q[:,:,i] = sd.differentiate_x(self.flux_qx[:,:,i]) \
                              + sd.differentiate_y(self.flux_qy[:,:,i])

    def __compute_source__(self):
        self.source[:,:,1] = self.g*self.q[:,:,0]*self.background_depth_xderiv \
                           - self.f*self.q[:,:,2]

        self.source[:,:,2] = self.g*self.q[:,:,0]*self.background_depth_yderiv \
                           + self.f*self.q[:,:,1]


    def __load_spatial_discretization__(self, spatial_discretization_name):
        mod = __import__('spatial_discretizations.',
                         spatial_discretization_name,
                         fromlist=[str(spatial_discretization_name)])
        submod = getattr(mod, spatial_discretization_name)
        return getattr(submod, spatial_discretization_name)


    def __load_time_stepper__(self, time_stepper_name):
        mod = __import__('time_steppers.',
                         time_stepper_name,
                         fromlist=[str(time_stepper_name)])
        submod = getattr(mod, time_stepper_name)
        return getattr(submod, time_stepper_name)

    def __load_outputter__(self, outputter_name):
        mod = __import__('outputters.',
                         outputter_name,
                         fromlist=[str(outputter_name)])
        submod = getattr(mod, outputter_name)
        theclass = getattr(submod, outputter_name)
        return theclass


    def __initialize_storage__(self):
        nx = self.spatial_discretization.num_points_x
        ny = self.spatial_discretization.num_points_y
        nf = SW_Solver.NUM_FIELDS
             
        self.q = np.zeros([ny, nx, nf])
        self.rhs_q = np.zeros([ny, nx, nf])
        self.temp = np.zeros([ny, nx, nf])
        self.flux_qx = np.zeros([ny, nx, nf])
        self.flux_qy = np.zeros([ny, nx, nf])
        self.div_flux_q = np.zeros([ny, nx, nf])
        self.source = np.zeros([ny, nx, nf])

