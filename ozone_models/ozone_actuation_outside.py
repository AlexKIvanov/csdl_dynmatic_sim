import numpy as np
import csdl
from eom_model import OzoneEulerFlatEarth6DoF
from tiltwing_mass_properties_model import tiltwing_mass_properties_model

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
        Fx = self.create_input('Fx', val=np.zeros((n,1)))
        Fy = self.create_input('Fy', val=np.zeros((n,1)))
        Fz = self.create_input('Fz', val=np.zeros((n,1)))

        # Create inputs for the moments
        Mx = self.create_input('Mx', val=np.zeros((n,1)))
        My = self.create_input('My', val=np.zeros((n,1)))
        Mz = self.create_input('Mz', val=np.zeros((n,1)))

        # Call mass properties model and promote everything 
        tilt_wing_mass_prop = tiltwing_mass_properties_model()
        self.add(tilt_wing_mass_prop, 'tiltwing_mass_prop')


        # Call the Ozone euler model
        eom_model = OzoneEulerFlatEarth6DoF(
            num_nodes = n,
        )
        self.add(eom_model, 'eom_model')


