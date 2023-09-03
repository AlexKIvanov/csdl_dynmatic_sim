import csdl
import numpy.testing
from fluids import atmosphere
from csdl_om import Simulator


class AtmosphereModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('name', types=str, default='ATMOSPHERE_1976')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        Z = self.declare_variable('altitude', shape=(num_nodes, ), val=0., units='m')

        H_std = [0.0, 11E3, 20E3, 32E3, 47E3, 51E3, 71E3, 84852.0]
        T_grad = [-6.5E-3, 0.0, 1E-3, 2.8E-3, 0.0, -2.8E-3, -2E-3, 0.0]
        T_std = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
        P_std = [101325, 22632.06397346291, 5474.8886696777745, 868.0186847552279,
                 110.90630555496608, 66.93887311868738, 3.956420428040732,
                 0.3733835899762159]

        r0 = 6356766.0
        P0 = 101325.0
        M0 = 28.9644
        g0 = 9.80665
        gamma = 1.400

        H = r0 * Z / (r0 + Z)

        T_layer = T_std[0]
        T_increase = T_grad[0]
        P_layer = P_std[0]
        H_layer = H_std[0]

        H_above_layer = H - H_layer

        # Temperature
        T = T_layer + T_increase * H_above_layer

        # Pressure
        R = 8314.32
        P = P_layer * (T_layer / T) ** (g0 * M0 / (R * T_increase))

        # Density
        rho = P * 0.00348367635597379 / T

        # Speed of sound
        a = (gamma*R/M0 * T)**0.5

        # Viscosity
        mu = 1.458E-6 * T * T**0.5 / (T + 110.4)

        # Thermal conductivity
        k = 2.64638E-3 * T * T**0.5 / (T + 245.4 * csdl.exp(-27.63102111592855 / T))

        # Acceleration due to gravity
        x0 = (r0 / (r0 + Z))
        gh = g0*x0*x0

        self.register_output('temperature', T)
        self.register_output('pressure', P)
        self.register_output('density', rho)
        self.register_output('dynamic_viscosity', mu)
        self.register_output('speed_of_sound', a)
        self.register_output('thermal_conductivity', k)
        self.register_output('acc_gravity', gh)


if __name__ == "__main__":
    atmos_model = AtmosphereModel(
        num_nodes=1,
    )
    sim = Simulator(atmos_model)

    # Test 1
    h = 1000
    atmos1_package = atmosphere.ATMOSPHERE_1976(Z=h)
    sim['altitude'] = h
    sim.run()
    numpy.testing.assert_almost_equal(actual=sim['acc_gravity'], desired=atmos1_package.g)
    numpy.testing.assert_almost_equal(actual=sim['density'], desired=atmos1_package.rho)
    numpy.testing.assert_almost_equal(actual=sim['speed_of_sound'], desired=atmos1_package.v_sonic)

    # Test 2
    h = 10668
    atmos2_package = atmosphere.ATMOSPHERE_1976(Z=h)
    sim['altitude'] = h
    sim.run()
    numpy.testing.assert_almost_equal(actual=sim['acc_gravity'], desired=atmos2_package.g)
    numpy.testing.assert_almost_equal(actual=sim['density'], desired=atmos2_package.rho)
    numpy.testing.assert_almost_equal(actual=sim['speed_of_sound'], desired=atmos2_package.v_sonic)
