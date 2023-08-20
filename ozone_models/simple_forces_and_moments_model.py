import numpy as np
import csdl


class simpleForcesAndMoments(csdl.Model):
    '''
    This model should receive the thrust vector origin, the thrust vector, and the force 
    that corresponds to that thrust vector. 

    Inputs:
        - thrust origin (nt,3)
        - thrust vector (nt,3)
        - thrust value  (nt,3) <---- This is a single scalar value that is expanded across all axes
    '''
    
    def initialize(self):
        self.parameters.declare('nt')
        self.parameters.declare('front_thrust_vector_dict')
        self.parameters.declare('rear_thrust_vector_dict')
        self.parameters.declare('thrust_dict')
        self.parameters.declare('refPt')

    def define(self):
        nt                         = self.parameters['nt']
        refPt                      = self.parameters['refPt']
        thrust_dict                = self.parameters['thrust_dict']
        front_thrust_vector_dict   = self.parameters['front_thrust_vector_dict']
        rear_thrust_vector_dict    = self.parameters['rear_thrust_vector_dict']
        thrust_vector_dict         = {**front_thrust_vector_dict, **rear_thrust_vector_dict}

        refPt         = self.declare_variable('refPt', val=refPt)
        refPtExpanded = csdl.expand(refPt, (nt,3), 'i->ji')

        Fx = self.create_output('Fx', shape=(nt,1))
        Fy = self.create_output('Fy', shape=(nt,1))
        Fz = self.create_output('Fz', shape=(nt,1))

        Mx = self.create_output('Mx', shape=(nt,1))
        My = self.create_output('My', shape=(nt,1))
        Mz = self.create_output('Mz', shape=(nt,1))
        
        



        for key, val in thrust_vector_dict.items():
            thrust_origin_val = val[0]
            thrust_vector_val = val[1]

            tempThrust = self.declare_variable(key+'_thrust', shape=thrust_dict[key].shape)
            tempOrigin = self.declare_variable(key+'_origin_rotated', shape=thrust_origin_val.shape)
            tempVector = self.declare_variable(key+'_vector_rotated', shape=thrust_vector_val.shape)

            # Expand the thrust variable
            thrust_mult_thrust_vector = csdl.expand(tempThrust, (nt,3), 'i->ij') 

            # Multiply expanded thrust scaler with the thrust vector
            thrust_vector_mult = thrust_mult_thrust_vector * tempVector
            self.register_output(key+'_thrust_vector_mult', thrust_vector_mult)

            # Compute the moments produced by the thrust vector around the refPt
            thrust_moments = csdl.cross(tempOrigin-refPtExpanded, thrust_vector_mult, axis=1)
            self.register_output(key+'_moments', thrust_moments)
