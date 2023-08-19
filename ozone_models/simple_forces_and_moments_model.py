import numpy as np
import csdl


class simpleForcesAndMoments(csdl.model):
    '''
    This model should receive the thrust vector origin, the thrust vector, and the force 
    that corresponds to that thrust vector. 
    '''
    
    def initialize(self):
        self.parameters.declare('front_thrust_vector_dict')
        self.parameters.declare('rear_thrust_vector_dict')
        self.parameters.declare('thrust_dict')
        self.parameters.declare('refPt')

    def define(self):
        refPt                      = self.parameters['refPt']
        thrust_dict                = self.parameters['thrust_dict']
        front_thrust_vector_dict   = self.parameters['front_thrust_vector_dict']
        rear_thrust_vector_dict    = self.parameters['rear_thrust_vector_dict']
        thrust_vector_dict         = {**front_thrust_vector_dict, **rear_thrust_vector_dict}

        for key, val in thrust_vector_dict.items():
            thrust_origin_val = val[0]
            thrust_vector_val = val[1]

            tempThrust = self.declare_variable(key+'_thrust', shape=thrust_dict[key].shape)
            tempOrigin = self.declare_variable(key+'_origin', shape=thrust_origin_val.shape)
            tempVector = self.declare_variable(key+'_vector', shape=thrust_vector_val.shape)
