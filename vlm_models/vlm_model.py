import csdl

from VLM_package.vlm_solver import VLMSolverModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
from pip import main
from numpy import arange
import numpy as np
from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

from fluids import atmosphere as atmosphere


class VLMAerodynamicsModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)

        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0.25)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m', values=['m', 'ft'])
        # self.parameters.declare('cl0', default=[0.6,0.6,0,0])
        self.parameters.declare('cl0', default=[0.0,0.0,0,0])


    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        mesh_unit = self.parameters['mesh_unit']

        free_stream_velocities = self.parameters['free_stream_velocities']

        eval_pts_option = self.parameters['eval_pts_option']

        eval_pts_location = self.parameters['eval_pts_location']
        sprs = self.parameters['sprs']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']

        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]

            self.declare_variable(surface_names[i], shape=surface_shapes[i])

        # rho = self.declare_variable('rho', shape=(num_nodes, 1))
        rho = self.create_input('rho', val=np.ones((num_nodes, 1)) * 1.1)

        eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3)
                           for x in surface_shapes]
        # coeffs_aoa = [(0.535, 0.091), (0.535, 0.091), (0.535, 0.091),
        #               (0.535, 0.091)]
        # coeffs_cd = [(0.00695, 1.297e-4, 1.466e-4),
        #              (0.00695, 1.297e-4, 1.466e-4),
        #              (0.00695, 1.297e-4, 1.466e-4),
        #              (0.00695, 1.297e-4, 1.466e-4)]

        submodel = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes,
            eval_pts_location=0.25,
            eval_pts_shapes=eval_pts_shapes,
            AcStates=
            'dummy',  # this is not used, just to make sure the inputs are what caddee needs
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=self.parameters['cl0'],
        )

        self.add(submodel, 'VLMSolverModel')