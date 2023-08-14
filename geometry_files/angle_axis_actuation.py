import numpy as np 
import csdl

class AngleAxisActuation(csdl.Model):
    '''
    This model rotates points based on the angle and axis provided. 

    The rotations are completed using quaternions.

    The inputs and outputs of this model should NOT be promoted. 

    PARAMETERS:
        - number of timesteps
        - number of thrust vector origin pairs 
        - number of meshes 

    INPUTS: 
        - normalized axis vector
        - starting point from which the axis vector originates
        - points which 

    '''
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_thrust_vector_origin_pairs')

    def define(self):

        n       = self.parameters['num_nodes'] 
        n_pairs = self.parameters['num_thrust_vector_origin_pairs']

        axis = self.declare_variable('axis', shape=(3,))
