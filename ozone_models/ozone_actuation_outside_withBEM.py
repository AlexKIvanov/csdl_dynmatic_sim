import numpy as np
import csdl

from ozone_models.eom_model import OzoneEulerFlatEarth6DoF
from ozone_models.tiltwing_mass_properties_model import tiltwing_mass_properties_model
from ozone_models.inertial_loads_model import InertialLoadsModel
from rotor_models.bem_model import BEMModel
# from run_scripts.tilt_wing_actuation_outside_ozone_unconstrained_withBEM import params_dict, omega, thrust_vector_dict

'''

This Ozone accepts the actuation of the tiltwing from an external model. 

This model assumes that the optimizer can choose the values of thrust that would optimally achieve the objective function. 

'''

class ODESystemModel(csdl.Model):
    def initialize(self):
        # This line is needed no matter what. Don't worry about it
        self.parameters.declare('num_nodes')

        self.parameters.declare('propeller_dict')
        self.parameters.declare('thrust_vector_dict')
        self.parameters.declare('omega')

    def define(self):
        # This line is needed no matter what. Don't worry about it
        n = self.parameters['num_nodes']   

        # Importing the necessary information for the solvers
        propeller_dict = self.parameters['propeller_dict']
        thrust_vector_dict = self.parameters['thrust_vector_dict']
        omega = self.parameters['omega']

        # Create inputs for every state 
        # States below are in the inertially fixed NED coordinate frame 
        x     = self.create_input('x', val=np.zeros((n,1)))
        y     = self.create_input('y', val=np.zeros((n,1)))
        z     = self.create_input('z', val=np.zeros((n,1)))
        u     = self.create_input('u', val=np.zeros((n,1)))
        v     = self.create_input('v', val=np.zeros((n,1)))
        w     = self.create_input('w', val=np.zeros((n,1)))

        # States below are euler angles
        theta = self.create_input('theta', val=np.zeros((n,1)))
        psi   = self.create_input('psi', val=np.zeros((n,1)))
        phi   = self.create_input('phi', val=np.zeros((n,1)))

        # States below are in the body coordinate frame
        p     = self.create_input('p', val=np.zeros((n,1)))
        q     = self.create_input('q', val=np.zeros((n,1)))
        r     = self.create_input('r', val=np.zeros((n,1)))

        # Receive RefPt
        refPt = self.create_input('refPt', shape=(3,))

        # Create inputs for the RPMS
        for key, value in omega.items():
            self.create_input(key, shape=(n,1))

        # Create inputs for thrust vector and thrust origin
        for key,value in thrust_vector_dict.items():
            thrust_origin_name = key + '_origin_rotated_METERS_NED_CG'
            thrust_vector_name = key + '_vector_rotated_NED'
            self.create_input(thrust_origin_name, shape=(n,3))
            self.create_input(thrust_vector_name, shape=(n,3))        

        # Create inputs for prop_radius, chord, twist
        for key, value in propeller_dict.items():
            if (key == 'prop_radius') or (key == 'chord') or (key == 'twist_cp'):
                self.create_input(key, val=value)

        # Call mass properties model and promote everything 
        tilt_wing_mass_prop = tiltwing_mass_properties_model()
        self.add(tilt_wing_mass_prop, 'tiltwing_mass_prop')

        # Call inertial loads
        inertial_loads = InertialLoadsModel(num_nodes=n)
        self.add(inertial_loads, 'inertial_loads_model')

        inertial_loads_F = self.declare_variable('inertial_loads_F', shape=(n, 3))
        inertial_loads_M = self.declare_variable('inertial_loads_M', shape=(n, 3))


        # Call BEM models for all the front and rear wing rotors
        for key, value in thrust_vector_dict.items():
            for key1, value1 in omega.items():
                if key in key1:
                    thrust_origin_name = key + '_origin_rotated_METERS_NED_CG'
                    thrust_vector_name = key + '_vector_rotated_NED'
                    solver_name = key+'_BemModel'
                    tempModel = BEMModel(
                        name = solver_name,
                        num_nodes = n,
                        num_radial = propeller_dict['num_radial'],
                        num_tangential = propeller_dict['num_tangential'],
                        airfoil = 'NACA_0012',
                        bem_mesh_list = [(thrust_origin_name, thrust_vector_name)],
                    )
                    self.add(tempModel, name=solver_name, promotes=[])

                    # Connect thrust origin/vector pointsets to the model
                    self.connect(thrust_origin_name, solver_name+ '.' + thrust_origin_name)
                    self.connect(thrust_vector_name, solver_name+ '.' + thrust_vector_name)

                    # Connect refPt 
                    self.connect('refPt', solver_name+'.ref_pt')

                    # Connect RPMs
                    self.connect(key1, solver_name+'.omega')

                    # Connect inputs for prop_radius, chord, twist
                    for tempKey, value in propeller_dict.items():
                        if (tempKey == 'prop_radius'): 
                            self.connect(tempKey, solver_name + '.propeller_radius')
                        
                        elif (tempKey == 'chord'):
                            self.connect(tempKey, solver_name + '.chord_profile')
                            
                        elif (tempKey == 'twist_cp'):
                            self.connect(tempKey, solver_name + '.pitch_cp')

                    # Connect bem states [z, u, v, w, p, q, r]
                    bem_states = ['z', 'u', 'v', 'w', 'p', 'q', 'r']
                    for bem_state in bem_states:
                        self.connect(bem_state, solver_name + '.' + bem_state)

        F_bem_total_presum = self.create_output('F_bem_total_presum', shape=(4, n, 3))
        M_bem_total_presum = self.create_output('M_bem_total_presum', shape=(4, n, 3))
        
        count = 0
        for key, value in thrust_vector_dict.items():
            F_solver_name = key+'_BemModel_F'
            M_solver_name = key+'_BemModel_M'
            solver_name = key+'_BemModel'

            F_temp = self.declare_variable(F_solver_name, shape=(n, 3))
            M_temp = self.declare_variable(M_solver_name, shape=(n, 3))

            self.connect(f'{solver_name}.{solver_name}_F', F_solver_name)
            self.connect(f'{solver_name}.{solver_name}_M', M_solver_name)
            
            F_bem_total_presum[count, :, 0] = csdl.expand(F_temp[:,0], (1,n,1), 'ij->kij')
            F_bem_total_presum[count, :, 1] = csdl.expand(F_temp[:,1], (1,n,1), 'ij->kij')
            F_bem_total_presum[count, :, 2] = csdl.expand(F_temp[:,2], (1,n,1), 'ij->kij')
            
            M_bem_total_presum[count, :, 0] = csdl.expand(M_temp[:,0], (1,n,1), 'ij->kij')
            M_bem_total_presum[count, :, 1] = csdl.expand(M_temp[:,1], (1,n,1), 'ij->kij')
            M_bem_total_presum[count, :, 2] = csdl.expand(M_temp[:,2], (1,n,1), 'ij->kij')

            count += 1

        F_bem_total = csdl.sum(F_bem_total_presum, axes=(0,))
        M_bem_total = csdl.sum(M_bem_total_presum, axes=(0,))

        self.register_output('F_bem_total', F_bem_total)
        self.register_output('M_bem_total', M_bem_total)
        
        total_Fx = inertial_loads_F[:,0] + F_bem_total[:,0]
        total_Fy = inertial_loads_F[:,1] + F_bem_total[:,1]
        total_Fz = inertial_loads_F[:,2] + F_bem_total[:,2]

        total_Mx = inertial_loads_M[:,0] + M_bem_total[:,0]
        total_My = inertial_loads_M[:,1] + M_bem_total[:,1]
        total_Mz = inertial_loads_M[:,2] + M_bem_total[:,2]
        
        self.register_output('total_Fx', total_Fx)
        self.register_output('total_Fy', total_Fy)
        self.register_output('total_Fz', total_Fz)

        self.register_output('total_Mx', total_Mx)
        self.register_output('total_My', total_My)
        self.register_output('total_Mz', total_Mz)


        # Call the Ozone euler model
        eom_model = OzoneEulerFlatEarth6DoF(
            num_nodes = n,
        )
        self.add(eom_model, 'eom_model')


