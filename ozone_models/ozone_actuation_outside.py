import numpy as np
import csdl

from ozone_models.eom_model import OzoneEulerFlatEarth6DoF
from ozone_models.tiltwing_mass_properties_model import tiltwing_mass_properties_model
from ozone_models.inertial_loads_model import InertialLoadsModel

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
        n = self.parameters['num_nodes']   

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

        # Create inputs for the Forces
        Fx = self.create_input('forces_moments_Fx', val=np.zeros((n,1)))
        Fy = self.create_input('forces_moments_Fy', val=np.zeros((n,1)))
        Fz = self.create_input('forces_moments_Fz', val=np.zeros((n,1)))

        # Create inputs for the moments
        Mx = self.create_input('forces_moments_Mx', val=np.zeros((n,1)))
        My = self.create_input('forces_moments_My', val=np.zeros((n,1)))
        Mz = self.create_input('forces_moments_Mz', val=np.zeros((n,1)))

        # Receive RefPt
        refPt = self.create_input('refPt', val=np.zeros((3,)))

        # Call mass properties model and promote everything 
        tilt_wing_mass_prop = tiltwing_mass_properties_model()
        self.add(tilt_wing_mass_prop, 'tiltwing_mass_prop')

        # Call inertial loads
        inertial_loads = InertialLoadsModel(num_nodes=n)
        self.add(inertial_loads, 'inertial_loads_model')

        inertial_loads_F = self.declare_variable('inertial_loads_F', shape=(n, 3))
        inertial_loads_M = self.declare_variable('inertial_loads_M', shape=(n, 3))
        
        total_Fx = inertial_loads_F[:,0] + Fx
        total_Fy = inertial_loads_F[:,1] + Fy
        total_Fz = inertial_loads_F[:,2] + Fz

        total_Mx = inertial_loads_M[:,0] + Mx
        total_My = inertial_loads_M[:,1] + My
        total_Mz = inertial_loads_M[:,2] + Mz
        
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


