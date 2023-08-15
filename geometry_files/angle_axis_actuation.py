import numpy as np 
import csdl

class AngleAxisActuation(csdl.Model):
    '''
    This model rotates points based on the angle and axis provided. 

    The rotations are completed using quaternions.

    The inputs and outputs of this model should NOT be promoted. 

    PARAMETERS:
        - number of timesteps
        - thrust vector dictionary 
        - vlm mesh dictionary 
        - axis dictionary
        - actuation angle dictionary

    INPUTS: 
        - normalized axis vector
        - origin point
        - thrust origin point
        - thrust vector 
        - vlm mesh

    OUTPUTS:
        - actuated thrust origins
        - actuated thrust vectors
        - actuated meshes

    '''
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('thrust_vector_dict')
        self.parameters.declare('vlm_mesh_dict')
        self.parameters.declare('axis_dict')
        self.parameters.declare('actuation_angle_dict')

    def define(self):

        n                    = self.parameters['num_nodes'] 
        thrust_vector_dict   = self.parameters['thrust_vector_dict']
        vlm_mesh_dict        = self.parameters['vlm_mesh_dict']
        axis_dict            = self.parameters['axis_dict']
        actuation_angle_dict = self.parameters['actuation_angle_dict']

        
