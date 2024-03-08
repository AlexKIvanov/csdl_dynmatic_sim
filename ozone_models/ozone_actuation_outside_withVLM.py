import numpy as np
import csdl

from ozone_models.eom_model import OzoneEulerFlatEarth6DoF
from ozone_models.tiltwing_mass_properties_model import tiltwing_mass_properties_model
from ozone_models.inertial_loads_model import InertialLoadsModel
from vlm_models.vlm_model import VLMAerodynamicsModel

# from run_scripts.tilt_wing_actuation_outside_ozone_unconstrained_withBEM import params_dict, omega, thrust_vector_dict

'''

This Ozone accepts the actuation of the tiltwing from an external model. 

This model assumes that the optimizer can choose the values of thrust that would optimally achieve the objective function. 

'''

class ODESystemModel(csdl.Model):
    def initialize(self):
        # This line is needed no matter what. Don't worry about it
        self.parameters.declare('num_nodes')

    def define(self):
        # This line is needed no matter what. Don't worry about it
        n             = self.parameters['num_nodes']   

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

        # Define gamma, phiw, psiw
        gamma = self.create_input('gamma', val=np.zeros((n,1)))
        psiw = self.create_input('psiw', val=np.zeros((n,1)))
        self.create_input('phiw', val=np.zeros((n,1)))

        # Create inputs for the Forces
        Fx = self.create_input('forces_moments_Fx', val=np.zeros((n,1)))
        Fy = self.create_input('forces_moments_Fy', val=np.zeros((n,1)))
        Fz = self.create_input('forces_moments_Fz', val=np.zeros((n,1)))

        # Create inputs for the moments
        Mx = self.create_input('forces_moments_Mx', val=np.zeros((n,1)))
        My = self.create_input('forces_moments_My', val=np.zeros((n,1)))
        Mz = self.create_input('forces_moments_Mz', val=np.zeros((n,1)))

        # Receive RefPt
        refPt = self.create_input('refPt', shape=(3,))

        self.print_var(refPt)


        # Receive vlm meshes
        front_wing_mesh = self.create_input('front_wing_vlm_rotated', shape=(n,3,3,3))
        rear_wing_mesh  = self.create_input('rear_wing_vlm_rotated', shape=(n,3,3,3))

        self.print_var(front_wing_mesh)
        self.print_var(rear_wing_mesh)

        vlm_mesh_shapes = [(3,3,3), (3,3,3)]
        vlm_mesh_names  = ['front_wing_vlm_rotated', 'rear_wing_vlm_rotated']

        ode_surface_shapes = [(n, ) + item for item in vlm_mesh_shapes]

        vlm_u = self.create_output('vlm_u', shape=(n,1))
        vlm_u[:,0] = -u

        vlm_w = self.create_output('vlm_w', shape=(n,1))
        vlm_w[:,0] = -w

        # Make sure to specify the correct airfoil
        vlm_model = VLMAerodynamicsModel(
            surface_names = vlm_mesh_names,
            surface_shapes = ode_surface_shapes,
            num_nodes = n,
            mesh_unit='m'
        )

        self.add(vlm_model, name='ozone_vlm_model', promotes=[])

        # Connecting the reference point to the VLM
        self.connect('refPt', 'ozone_vlm_model.evaluation_pt')

        # Connecting the states to the VLM
        vlm_states = ['z', 'v', 'p', 'q', 'r', 'gamma', 'theta', 'psi', 'psiw', 'phi']

        for vlm_state in vlm_states:
            self.connect(vlm_state, 'ozone_vlm_model.' + vlm_state)

        self.connect('vlm_w', 'ozone_vlm_model.w')
        self.connect('vlm_u', 'ozone_vlm_model.u')
        

        # Connect surfaces into VLM
        self.connect('front_wing_vlm_rotated', 'ozone_vlm_model.front_wing_vlm_rotated')
        self.connect('rear_wing_vlm_rotated',  'ozone_vlm_model.rear_wing_vlm_rotated')
        
        # Call mass properties model and promote everything 
        tilt_wing_mass_prop = tiltwing_mass_properties_model()
        self.add(tilt_wing_mass_prop, 'tiltwing_mass_prop')

        # Call inertial loads
        inertial_loads = InertialLoadsModel(num_nodes=n)
        self.add(inertial_loads, 'inertial_loads_model')

        inertial_loads_F = self.declare_variable('inertial_loads_F', shape=(n, 3))
        inertial_loads_M = self.declare_variable('inertial_loads_M', shape=(n, 3))

        # Extract forces and moments from VLM
        vlm_f_vlmFrame = self.declare_variable('vlm_F_vlmFrame', shape=(n,3))
        vlm_m_vlmFrame = self.declare_variable('vlm_M_vlmFrame', shape=(n,3))

        self.connect('ozone_vlm_model.F', 'vlm_F_vlmFrame')
        self.connect('ozone_vlm_model.M', 'vlm_M_vlmFrame')

        # Define rotation matrix about X-axis to transform vlm frame to NED
        vlm2ned_mat = np.array([[1, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
        vlm2ned = self.create_input('vlm2ned', val=vlm2ned_mat)

        vlm_f      = self.create_output('vlm_f', shape=(n,3))
        vlm_f[:,0] = -vlm_f_vlmFrame[:,0]   # Drag is negative in NED
        vlm_f[:,1] = vlm_f_vlmFrame[:,1]
        vlm_f[:,2] = -vlm_f_vlmFrame[:,2]   # Lift is negative in NED
         
        vlm_m = self.create_output('vlm_m', shape=(n,3))
        vlm_m[:,0] = -vlm_m_vlmFrame[:,0]
        vlm_m[:,1] = vlm_m_vlmFrame[:,1]
        vlm_m[:,2] = -vlm_m_vlmFrame[:,2]

        self.print_var(vlm_f_vlmFrame)
        self.print_var(vlm_m_vlmFrame)

        # Extract forces and moments from inertial loads model
        inertial_loads_F = self.declare_variable('inertial_loads_F', shape=(n, 3))
        inertial_loads_M = self.declare_variable('inertial_loads_M', shape=(n, 3))

        # self.print_var(vlm_f)
        self.print_var(inertial_loads_F)
        self.print_var(inertial_loads_M)
        self.print_var(Fx)
        self.print_var(Fz)

        # Separate forces and moments in X,Y,Z components
        total_Fx = vlm_f[:,0] + inertial_loads_F[:,0] - Fx[:,0]
        total_Fy = vlm_f[:,1] + inertial_loads_F[:,1] + Fy[:,0]
        total_Fz = vlm_f[:,2] + inertial_loads_F[:,2] - Fz[:,0]

        total_Mx = vlm_m[:,0] + inertial_loads_M[:,0] + Mx[:,0]
        total_My = vlm_m[:,1] + inertial_loads_M[:,1] + My[:,0]
        total_Mz = vlm_m[:,2] + inertial_loads_M[:,2] + Mz[:,0]

        self.print_var(total_Fx)
        self.print_var(total_Fy)
        self.print_var(total_Fz)

        self.print_var(total_Mx)
        self.print_var(total_My)
        self.print_var(total_Mz)
         
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