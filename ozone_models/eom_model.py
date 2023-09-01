
import numpy as np
import csdl

class OzoneEulerFlatEarth6DoF(csdl.Model):
    eom_model_name = 'OzoneEulerEoM'
    """
    Euler flat earth equations: linear momentum equations, angular momentum
    equations, angular kinematic equations, linear kinematic equations.

    state_vector : array_like, shape(9)
        Current value of absolute velocity and angular velocity, both
        expressed in body axes, euler angles and position in Earth axis.
        (u, v, w, p, q, r, theta, phi, psi, x, y, z)
         (m/s, m/s, m/s, rad/s, rad/s rad/s, rad, rad, rad, m, m ,m).
    mass : float
        Current mass of the aircraft (kg).
    inertia : array_like, shape(3, 3)
        3x3 tensor of inertia of the aircraft (kg * m2)
        Current equations assume that the aircraft has a symmetry plane
        (x_b - z_b), thus J_xy and J_yz must be null.
    forces : array_like, shape(3)
        3 dimensional vector containing the total total_forces (including
        gravity) in x_b, y_b, z_b axes (N).
    moments : array_like, shape(3)
        3 dimensional vector containing the total total_moments in x_b,
        y_b, z_b axes (N·m).

    dstate_dt : array_like, shape(9)
        Derivative with respect to time of the state vector.
        Current value of absolute acceleration and angular acceleration,
        both expressed in body axes, Euler angles derivatives and velocity
        with respect to Earth Axis.
        (du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dtheta_dt, dphi_dt,
        dpsi_dt, dx_dt, dy_dt, dz_dt)
        (m/s² , m/s², m/s², rad/s², rad/s², rad/s², rad/s, rad/s, rad/s,
        m/s, m/s, m/s).

    References
    ----------
    .. [1] B. Etkin, "Dynamics of Atmospheric Flight", Courier Corporation,
        p. 149 (5.8 The Flat-Earth Approximation), 2012.

    .. [2] M. A. Gómez Tierno y M. Pérez Cortés, "Mecánica del Vuelo",
        Garceta Grupo Editorial, pp.18-25 (Tema 2: Ecuaciones Generales del
        Moviemiento), 2012.

    """
    def initialize(self):
        self.parameters.declare(name='name', default=self.eom_model_name)
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        num_nodes = self.parameters['num_nodes']

        # region Inputs
        # Loads
        Fx = self.declare_variable(name='total_Fx', shape=(num_nodes))
        Fy = self.declare_variable(name='total_Fy', shape=(num_nodes))
        Fz = self.declare_variable(name='total_Fz', shape=(num_nodes))
        L = self.declare_variable(name='total_Mx', shape=(num_nodes))
        M = self.declare_variable(name='total_My', shape=(num_nodes))
        N = self.declare_variable(name='total_Mz', shape=(num_nodes))

        # Mass properties
        mass = self.declare_variable(
            name='total_mass',
            shape=(1, ), units='kg')
        Ixx = self.declare_variable(
            name='ixx',
            shape=(1, ), units='kg*m**2')
        Iyy = self.declare_variable(
            name='iyy',
            shape=(1, ), units='kg*m**2')
        Izz = self.declare_variable(
            name='izz',
            shape=(1, ), units='kg*m**2')
        Ixz = self.declare_variable(
            name='ixz',
            shape=(1, ), units='kg*m**2')

        # State
        u = self.declare_variable(name='u', shape=(num_nodes))
        v = self.declare_variable(name='v', shape=(num_nodes))
        w = self.declare_variable(name='w', shape=(num_nodes))
        p = self.declare_variable(name='p', shape=(num_nodes))
        q = self.declare_variable(name='q', shape=(num_nodes))
        r = self.declare_variable(name='r', shape=(num_nodes))
        phi = self.declare_variable(name='phi', shape=(num_nodes))
        theta = self.declare_variable(name='theta', shape=(num_nodes))
        psi = self.declare_variable(name='psi', shape=(num_nodes))
        x = self.declare_variable(name='x', shape=(num_nodes))
        y = self.declare_variable(name='y', shape=(num_nodes))
        z = self.declare_variable(name='z', shape=(num_nodes))
        # endregion

        # region Calculations
        m = csdl.expand(var=mass, shape=(num_nodes))
        Ix = csdl.expand(var=Ixx, shape=(num_nodes))
        Iy = csdl.expand(var=Iyy, shape=(num_nodes))
        Iz = csdl.expand(var=Izz, shape=(num_nodes))
        Jxz = csdl.expand(var=Ixz, shape=(num_nodes))

        # Linear momentum equations
        du_dt = Fx / m + r * v - q * w + x * 0.
        dv_dt = (Fy / m) * 0. - r * u * 0. + p * w * 0. + y * 0.
        dw_dt = Fz / m + q * u - p * v + z * 0.

        # Angular momentum equations
        dp_dt = ((L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
                 p * q * Jxz * (Ix + Iz - Iy)) * 0.) / (Ix * Iz - Jxz ** 2)
        dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
        dr_dt = ((L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
                 q * r * Jxz * (Iz + Ix - Iy)) * 0.) / (Ix * Iz - Jxz ** 2)

        # Angular Kinematic equations
        dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
        dphi_dt = (p + (q * csdl.sin(phi) + r * csdl.cos(phi)) * csdl.tan(theta)) * 0.0 
        dpsi_dt = ((q * csdl.sin(phi) + r * csdl.cos(phi)) / csdl.cos(theta)) * 0.0  

        # Linear kinematic equations
        dx_dt = (csdl.cos(theta) * csdl.cos(psi) * u +
                 (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) * v +
                 (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi)) * w)
        dy_dt = (csdl.cos(theta) * csdl.sin(psi) * u +
                 (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) * v +
                 (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi)) * w) * 0.
        dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(
            phi) * csdl.cos(theta)
        # endregion


        # registering outputs of the Ozone Model
        self.register_output('du_dt', du_dt)
        self.register_output('dv_dt', dv_dt)
        self.register_output('dw_dt', dw_dt)

        self.register_output('dp_dt', dp_dt)
        self.register_output('dq_dt', dq_dt)
        self.register_output('dr_dt', dr_dt)

        self.register_output('dtheta_dt', dtheta_dt)
        self.register_output('dphi_dt', dphi_dt)
        self.register_output('dpsi_dt', dpsi_dt)

        self.register_output('dx_dt', dx_dt)
        self.register_output('dy_dt', dy_dt)
        self.register_output('dz_dt', dz_dt)
        
        return