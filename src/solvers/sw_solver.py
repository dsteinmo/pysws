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

        self.phys_params = config['phys_params']
        self.spatial_params = config['spatial_params']
        self.temporal_params = config['temporal_params']
        self.outputters = config['outputters']

        spatial_discretization_name = \
                  self.spatial_params['discretization_method']
        print "Loading spatial discretization: ", spatial_discretization_name
        
        SpatialDiscretization = self.__load_spatial_discretization__(
                                       spatial_discretization_name)

        # Instantiate spatial discretization.
        self.spatial_discretization = SpatialDiscretization(
                                                    self.spatial_params)

        self.__initialize_storage__()

    def initialize_fields(self, eta_initial, u_initial, v_initial,
                           background_depth):
        print "foo"

    def solve(self):
        print "Solving things!"

    def __load_spatial_discretization__(self, spatial_discretization_name):
        mod = __import__('spatial_discretizations.',
                         spatial_discretization_name,
                         fromlist=[str(spatial_discretization_name)])
        submod = getattr(mod, spatial_discretization_name)
        return getattr(submod, spatial_discretization_name)

    def __initialize_storage__(self):
        nx = self.spatial_discretization.num_points_x
        ny = self.spatial_discretization.num_points_y
        nf = SW_Solver.NUM_FIELDS
             
        self.q = np.zeros([ny, nx, nf])
        self.rhs_q = np.zeros([ny, nx, nf])
        self.res_q = np.zeros([ny, nx, nf])
        self.flux_qx = np.zeros([ny, nx, nf])
        self.flux_qy = np.zeros([ny, nx, nf])
        self.div_q = np.zeros([ny, nx, nf])
        self.source = np.zeros([ny, nx, nf])

