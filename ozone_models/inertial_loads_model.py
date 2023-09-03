import csdl
import numpy as np
from ozone_models.atmosphere_model import AtmosphereModel


# noinspection PyPep8Naming
class InertialLoadsModel(csdl.Model):
    def initialize(self):
        self.parameters.declare(name='name', default='mp')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        num_nodes = self.parameters['num_nodes']
        name = self.parameters['name']

        # Inputs constant across conditions (segments)
        cgx = self.declare_variable('cgx', shape=(1, ), units='m')
        cgy = self.declare_variable('cgy', shape=(1, ), units='m')
        cgz = self.declare_variable('cgz', shape=(1, ), units='m')
        m = self.declare_variable('m', shape=(1, ), units='kg')
        ref_pt = self.declare_variable(name='ref_pt', shape=(3, ), units='m')

        mass = csdl.expand(var=m, shape=(num_nodes, 1))

        # self.print_var(mass)

        # Inputs changing across conditions (segments)
        th = self.declare_variable('Theta', shape=(num_nodes, 1), units='rad')
        ph = self.declare_variable('Phi',
                                   shape=(num_nodes, 1),
                                   units='rad',
                                   val=0.)
        z = self.declare_variable('z', shape=(num_nodes, 1), units='m', val=0.)

        testTH = th * np.ones((num_nodes, 1))
        testPH = ph * np.ones((num_nodes, 1))
        testRefPt = ref_pt * np.ones((3,))

        self.register_output('testTH', testTH)
        self.register_output('testPH', testPH)
        self.register_output('testRefPt', testRefPt)
        
        # self.print_var(testTH)
        # self.print_var(testPH)
        # self.print_var(testRefPt)

        cg = self.create_output(name='cg', shape=(3, ))
        cg[0] = cgx
        cg[1] = cgy
        cg[2] = cgz

        # self.print_var(cg)

        # region Atmosisa
        self.register_output(name='h_for_atmosisa', var=z + 0.)
        atmosisa = AtmosphereModel(num_nodes=num_nodes)

        self.add(submodel=atmosisa,
                 name='atmoshphere_model',
                 promotes=[])

        self.connect('h_for_atmosisa',
                     'atmoshphere_model.altitude')

        g = self.declare_variable(name='g', shape=(num_nodes, 1))
        self.connect('atmosphere_model.acc_gravity',
                     'g')

        # self.print_var(var=rho)
        # endregion

        F = self.create_output(name='F', shape=(num_nodes, 3))

        F[:, 0] = -mass * g * csdl.sin(th)
        F[:, 1] = mass * g * csdl.cos(th) * csdl.sin(ph)
        F[:, 2] = mass * g * csdl.cos(th) * csdl.cos(ph)

        r_vec = cg - ref_pt
        r_vec = csdl.reshape(r_vec, (1, 3))
        M = self.create_output(name='M', shape=(num_nodes, 3))
        for n in range(num_nodes):
            M[n, :] = csdl.cross(r_vec, F[n, :], axis=1)
        return


# endregion
