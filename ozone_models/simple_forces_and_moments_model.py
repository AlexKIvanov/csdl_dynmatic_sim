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

        dict_len = len(thrust_vector_dict)

        Fx = self.create_output('Fx', shape=(dict_len, nt,1))
        Fy = self.create_output('Fy', shape=(dict_len, nt,1))
        Fz = self.create_output('Fz', shape=(dict_len, nt,1))

        Mx = self.create_output('Mx', shape=(dict_len, nt,1))
        My = self.create_output('My', shape=(dict_len, nt,1))
        Mz = self.create_output('Mz', shape=(dict_len, nt,1))
        
        cnt = 0
        for key, val in thrust_vector_dict.items():
            thrust_origin_val = val[0]
            thrust_vector_val = val[1]

            tempThrust = self.declare_variable(key+'_NEWTONS_thrust', shape=thrust_dict[key+'_NEWTONS'].shape)
            tempOrigin = self.declare_variable(key+'_origin_rotated_METERS_NED_CG', shape=thrust_origin_val.shape)
            tempVector = self.declare_variable(key+'_vector_rotated_NED', shape=thrust_vector_val.shape)

            # Expand the thrust variable
            thrust_NEWTONS_expanded = csdl.expand(tempThrust, (nt,3), 'i->ij') 

            # Multiply expanded thrust scaler with the thrust vector
            thrust_vector_mult = thrust_NEWTONS_expanded * tempVector
            self.register_output(key+'_thrust_vector_mult_NEWTONS_NED_CG', thrust_vector_mult)

            self.print_var(thrust_vector_mult)
            self.print_var(tempOrigin)
            # Compute the moments produced by the thrust vector around the refPt
            thrust_moments = csdl.cross(tempOrigin, thrust_vector_mult, axis=1)
            self.register_output(key+'_moments_NEWTON_METER_NED_CG', thrust_moments)

            self.print_var(thrust_moments)

            Fx[cnt,:,0] = csdl.reshape(thrust_vector_mult[:,0], (1,nt,1))
            Fy[cnt,:,0] = csdl.reshape(thrust_vector_mult[:,1], (1,nt,1))
            Fz[cnt,:,0] = csdl.reshape(thrust_vector_mult[:,2], (1,nt,1))

            Mx[cnt,:,0] = csdl.reshape(thrust_moments[:,0], (1,nt,1))
            My[cnt,:,0] = csdl.reshape(thrust_moments[:,1], (1,nt,1))
            Mz[cnt,:,0] = csdl.reshape(thrust_moments[:,2], (1,nt,1))   
            cnt = cnt + 1

        
        summedFx = csdl.sum(Fx, axes=(0,))
        summedFy = csdl.sum(Fy, axes=(0,)) * np.zeros((nt,1))
        summedFz = csdl.sum(Fz, axes=(0,))

        summedMx = csdl.sum(Mx, axes=(0,))* np.zeros((nt,1))
        summedMy = csdl.sum(My, axes=(0,)) 
        summedMz = csdl.sum(Mz, axes=(0,))* np.zeros((nt,1))


        self.register_output('forces_moments_Fx', summedFx)
        self.register_output('forces_moments_Fy', summedFy)
        self.register_output('forces_moments_Fz', summedFz)

        self.register_output('forces_moments_Mx', summedMx)
        self.register_output('forces_moments_My', summedMy)
        self.register_output('forces_moments_Mz', summedMz)

        self.print_var(summedFx)
        self.print_var(summedFy)
        self.print_var(summedFz)

        self.print_var(summedMx)
        self.print_var(summedMy)
        self.print_var(summedMz)
        
                    
            
