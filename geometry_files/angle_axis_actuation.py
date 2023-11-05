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
        self.parameters.declare('axis_origin_point')

    def define(self):

        n                    = self.parameters['num_nodes'] 
        thrust_vector_dict   = self.parameters['thrust_vector_dict']
        vlm_mesh_dict        = self.parameters['vlm_mesh_dict']
        axis_dict            = self.parameters['axis_dict']
        actuation_angle_dict = self.parameters['actuation_angle_dict']
        axis_origin_point    = self.parameters['axis_origin_point']


        # Create the actuation angle input
        for key, value in actuation_angle_dict.items():
            temp_act_profile = np.reshape(value, (n,1))
            actuation_angle = self.declare_variable(key, shape=temp_act_profile.shape)

        # Create normalized axis inputs
        for key, value in axis_dict.items():
            axis_name = key
            x_axis_value = np.reshape(value[:,0], (n,1))                  # Should be a numpy array of shape (n,1)
            y_axis_value = np.reshape(value[:,1], (n,1))                  # Should be a numpy array of shape (n,1)
            z_axis_value = np.reshape(value[:,2], (n,1))                  # Should be a numpy array of shape (n,1)
            
            x_axis = self.create_input(key+'_x' , val=x_axis_value)
            y_axis = self.create_input(key+'_y' , val=y_axis_value)
            z_axis = self.create_input(key+'_z' , val=z_axis_value)

        # Check if the dictionary is empty
        if len(thrust_vector_dict) == 0:
            pass
        else:
            for key, value in thrust_vector_dict.items():
                thrust_axis_origin_pt = np.reshape(axis_origin_point, (n,3))

                thrust_origin_name = key + '_origin'
                thrust_origin_val = value[0]          # Should be a numpy array of shape (n,3)

                thrust_vector_name = key + '_vector'
                thrust_vector_val = value[1]          # Should be a numpy array of shape (n,3)

                axis_origin_pt = self.create_input(key+'_axis_origin_pt_METERS_VSP', val=thrust_axis_origin_pt)

                thrust_origin = self.create_input(thrust_origin_name, val=thrust_origin_val)
                thrust_vector = self.create_input(thrust_vector_name, val=thrust_vector_val)

                thrust_origin_translated = thrust_origin - axis_origin_pt
                thrust_vector_translated = thrust_vector

                quat_origin = self.create_output(thrust_origin_name + '_quat', shape=(n,) + (4,))
                quat_vector = self.create_output(thrust_vector_name + '_quat', shape=(n,) + (4,))

                quat_origin[:,0] = csdl.cos(actuation_angle / 2)
                quat_origin[:,1] = csdl.sin(actuation_angle / 2) * x_axis
                quat_origin[:,2] = csdl.sin(actuation_angle / 2) * y_axis
                quat_origin[:,3] = csdl.sin(actuation_angle / 2) * z_axis
                
                quat_vector[:,0] = csdl.cos(actuation_angle / 2)
                quat_vector[:,1] = csdl.sin(actuation_angle / 2) * x_axis
                quat_vector[:,2] = csdl.sin(actuation_angle / 2) * y_axis
                quat_vector[:,3] = csdl.sin(actuation_angle / 2) * z_axis

                translated_rotated_thrust_origin = csdl.quatrotvec(quat_origin, thrust_origin_translated)
                translated_rotated_thrust_vector = csdl.quatrotvec(quat_vector, thrust_vector_translated)

                rotated_thrust_origin = translated_rotated_thrust_origin + axis_origin_pt
                rotated_thrust_vector = translated_rotated_thrust_vector

                self.register_output(thrust_origin_name+'_rotated_METERS_NED_CG', rotated_thrust_origin)
                self.register_output(thrust_vector_name+'_rotated_NED', rotated_thrust_vector)
                


        # Check if the dictionary is empty
        if len(vlm_mesh_dict) == 0:
            pass
        else:
            for key, value in vlm_mesh_dict.items():
                vlm_mesh_name = key
                vlm_mesh_val = value                 # Should be a numpy array of shape (n,p,3)
                
                vlm_mesh = self.create_input(vlm_mesh_name, val=vlm_mesh_val)

        

            




