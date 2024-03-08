from email.policy import default
import numpy as np
from csdl import Model
import csdl

# from lsdo_rotor.rotor_parameters import RotorParameters
from lsdo_rotor.core.BEM.inputs.BEM_external_inputs_model import BEMExternalInputsModel
from lsdo_rotor.core.BEM.inputs.BEM_core_inputs_model import BEMCoreInputsModel
from lsdo_rotor.core.BEM.inputs.BEM_pre_process_model import BEMPreprocessModel
from lsdo_rotor.core.BEM.BEM_bracketed_search_model import BEMBracketedSearchGroup
from lsdo_rotor.core.BEM.BEM_prandtl_loss_factor_model import BEMPrandtlLossFactorModel
from lsdo_rotor.core.BEM.BEM_induced_velocity_model import BEMInducedVelocityModel
from lsdo_rotor.airfoil.BEM_airfoil_surrogate_model_group_2 import BEMAirfoilSurrogateModelGroup2

from lsdo_rotor.airfoil.get_surrogate_model import get_surrogate_model
from lsdo_rotor.core.BEM.functions.get_BEM_rotor_dictionary import get_BEM_rotor_dictionary

from lsdo_atmos.atmosphere_model import AtmosphereModel

from lsdo_rotor.core.BEM.functions.get_bspline_mtx import get_bspline_mtx
from lsdo_rotor.core.BEM.BEM_b_spline_comp import BsplineComp
from src.caddee.concept.geometry.geocore.mesh import BEMMesh


class BEMModel(Model):
    def initialize(self):
        self.parameters.declare(name='name')
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('num_radial', types=int, default=30)
        self.parameters.declare('num_tangential', types=int, default=30)
        self.parameters.declare(name='airfoil', default='NACA_0012')
        # ------
        self.parameters.declare('bem_mesh_list', types=list)
        # ------
        # self.parameters.declare('thrust_vector', types=np.ndarray)
        # self.parameters.declare('thrust_origin', types=np.ndarray)
        # self.parameters.declare('ref_pt', types=np.ndarray)
        # self.parameters.declare('thrust_vector', types=np.ndarray)
        self.parameters.declare('num_blades', types=int, default=5)

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']
        num_radial = self.parameters['num_radial']
        num_tangential = self.parameters['num_tangential']
        airfoil = self.parameters['airfoil']
        # shape = self.parameters['shape']

        # thrust_vector = self.parameters['thrust_vector']
        # thrust_origin = self.parameters['thrust_origin']
        # ref_pt = self.parameters['ref_pt']

        # thrust_vector = self.parameters['thrust_vector']
        # thrust_origin = self.parameters['thrust_origin']
        # ref_pt = self.parameters['ref_pt']

        # --------------------------------------------------------------------------- #
        bem_mesh_list = self.parameters['bem_mesh_list']
        BEM_pt_set_list = []
        T_v_name_list= []
        mm2ft = 304.8
        ft2m = 0.3048

        counter = 0
        for i in bem_mesh_list:
            BEM_dict = {
                'origin': self.declare_variable(i[0], shape=(num_nodes, 1, 3)),
                'vector': self.declare_variable(i[1], shape=(num_nodes, 1, 3))
            }
            thrust_origin_str = i[0] + '_thrust_origin'  #"% s" % counter
            thrust_vector_str = i[1] + '_thrust_vector'  #"% s" % counter
            BEM_pt_set_list.append(BEM_dict)
            T_o = 1 * (csdl.reshape(BEM_pt_set_list[counter]['origin'],new_shape=(num_nodes, 3))) * ft2m
            # self.print_var(T_o)
            self.register_output(thrust_origin_str, T_o)
            # self.print_var(T_o)
            T_v = csdl.reshape(BEM_pt_set_list[counter]['vector'], new_shape=(num_nodes, 3)) 
            T_v_mag = csdl.pnorm(T_v, axis=1)
            T_v_dir = -(T_v / csdl.expand(T_v_mag, (num_nodes, 3), 'i->ij'))

            # self.print_var(T_v)
            
            # self.print_var(BEM_pt_set_list[counter]['vector'])
            # self.print_var(T_v_dir)
            self.register_output(thrust_vector_str, T_v_dir)
            T_v_name_list.append(thrust_vector_str)
            counter += 1

        # ref_pt = self.declare_variable(name='ref_pt', shape=(num_nodes,3), units='m')

        num_blades = self.parameters['num_blades']
        shape = (num_nodes, num_radial, num_tangential)

        interp = get_surrogate_model(airfoil)
        rotor = get_BEM_rotor_dictionary(airfoil, interp)

        pitch_cp = self.declare_variable(name='pitch_cp', shape=(4, ),units='rad',val=np.linspace(50, 10, 4) * np.pi /180)
        pitch_A = get_bspline_mtx(4, num_radial, order=4)
        comp = csdl.custom(pitch_cp, op=BsplineComp(
            num_pt=num_radial,
            num_cp=4,
            in_name='pitch_cp',
            jac=pitch_A,
            out_name='twist_profile',
        ))
        self.register_output('twist_profile', comp)

        self.add(BEMExternalInputsModel(
            shape=shape,
            T_v_name_list=T_v_name_list,
        ), name='BEM_external_inputs_model')  #, promotes = ['*'])

        self.add(BEMCoreInputsModel(
            shape=shape,
        ), name='BEM_core_inputs_model')

        self.add(BEMPreprocessModel(
            shape=shape,
            num_blades=num_blades,
        ), name='BEM_pre_process_model')

        self.add(AtmosphereModel(
            shape=(num_nodes, 1), 
        ), name='atmosphere_model')

        chord = self.declare_variable('_chord', shape=shape)
        Vx = self.declare_variable('_axial_inflow_velocity', shape=shape)
        Vt = self.declare_variable('_tangential_inflow_velocity', shape=shape)
        W = (Vx**2 + Vt**2)**0.5
        rho = csdl.expand(self.declare_variable('density', shape=(num_nodes, )), shape, 'i->ijk')
        mu = csdl.expand(self.declare_variable('dynamic_viscosity', shape=(num_nodes, )), shape, 'i->ijk')
        Re = rho * W * chord / mu
        self.register_output('Re', Re)

        tempChord = self.register_output('tempChord', chord*np.ones(shape))
        # self.print_var(tempChord)

        # tempVx = self.register_output('tempVx', Vx*np.ones(shape))
        # self.print_var(tempVx)

        # tempVt = self.register_output('tempVt', Vt*np.ones(shape))
        # self.print_var(tempVt)

        tempW = self.register_output('tempW', W*np.ones(shape))
        # self.print_var(tempW)

        # temprho = self.register_output('temprho', rho*np.ones(shape))
        # self.print_var(temprho)

        tempmu = self.register_output('tempmu', mu*np.ones(shape))
        # self.print_var(tempmu)

        tempRe = self.register_output('tempRe', Re*np.ones(shape))
        # self.print_var(tempRe)

        self.add(BEMBracketedSearchGroup(
            rotor=rotor,
            shape=shape,
            num_blades=num_blades,
        ), name='phi_bracketed_search_group')  #, promotes = ['*'])

        phi = self.declare_variable('phi_distribution', shape=shape)
        twist = self.declare_variable('_pitch', shape=shape)
        alpha = twist - phi
        tempAoA = self.register_output('AoA', alpha)
        # self.print_var(tempAoA)

        airfoil_model_output_2 = csdl.custom(Re,alpha,chord,op=BEMAirfoilSurrogateModelGroup2(
            rotor=rotor,
            shape=shape,
        ))
        tempCl2 = self.register_output('Cl_2', airfoil_model_output_2[0])
        tempCd2 = self.register_output('Cd_2', airfoil_model_output_2[1])

        # self.print_var(tempCl2)
        # self.print_var(tempCd2)

        self.add(BEMPrandtlLossFactorModel(
            shape=shape,
            num_blades=num_blades,
        ), name='prandtl_loss_factor_model')  #, promotes = ['*'])

        self.add(BEMInducedVelocityModel(
            shape=shape,
            num_blades=num_blades,
        ), name='induced_velocity_model')  #, promotes = ['*'])

        # Post-Processing
        T = self.declare_variable('T', shape=(num_nodes, ))
        F = self.create_output(name+'_F', shape=(num_nodes, 3))
        M = self.create_output(name+'_M', shape=(num_nodes, 3))

        # self.print_var(T)

        ref_pt = self.declare_variable(name='ref_pt', shape=(1, 3), units='m')
        tempRefPt  = ref_pt * np.ones((1, 3))
        tempExpRef = self.register_output('expanded_reference', tempRefPt)
        # self.print_var(tempExpRef)
        # ref_pt_exp = csdl.expand(ref_pt,(num_nodes,3), 'j->ij')
        # loop over pt set list
        # thrust_origin_3 = self.declare_variable('thrust_origin_3', shape=(num_nodes,3), val=np.tile(np.array([[8.5,0,5]]), (num_nodes,1)))
        thrust_origin = T_o #self.declare_variable('thrust_origin_3',shape=(num_nodes, 3),val=np.tile(np.array([[0, 0, 0]]),(num_nodes, 1)))
        thrust_vector = T_v_dir #self.declare_variable('thrust_vector_3',shape=(num_nodes, 3),val=np.tile(np.array([[1, 0, 0]]),(num_nodes, 1)))
        
        # for i in range(num_nodes):
        #     F[i, :] = csdl.expand(T[i], (1, 3)) * thrust_vector[i, :]
        #     self.print_var(F)
        #     M[i, 0] = F[i, 2] * (thrust_origin[i, 1] - ref_pt[0, 1]) * np.zeros((1,))
        #     M[i, 1] = F[i, 2] * (thrust_origin[i, 0] - ref_pt[0, 0])
        #     M[i, 2] = F[i, 0] * (thrust_origin[i, 1] - ref_pt[0, 1]) * np.zeros((1,)) 
        u = self.declare_variable('u', shape=(num_nodes,1))
        v = self.declare_variable('v', shape=(num_nodes,1))
        w = self.declare_variable('w', shape=(num_nodes,1))

        temptempu = u * np.ones((num_nodes,1))
        temptempv = v * np.ones((num_nodes,1))
        temptempw = w * np.ones((num_nodes,1))

        temptempprintu = self.register_output('temptempU', temptempu)
        temptempprintv = self.register_output('temptempV', temptempv)
        temptempprintw = self.register_output('temptempW', temptempw)

        # self.print_var(temptempprintu)
        # self.print_var(temptempprintv)
        # self.print_var(temptempprintw)

        for i in range(num_nodes):
            print('T_v_name_list',T_v_name_list)
            # F[i, :] = csdl.expand(T[i], (1, 3)) * thrust_vector[i, :]
            F[i, 0] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 0] #- 9 * hub_drag[i,0]
            F[i, 1] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 1] * np.zeros((1,1))
            F[i, 2] = csdl.reshape(T[i], (1, 1)) * thrust_vector[i, 2]
            
            
            # self.print_var(F)
            # self.print_var(thrust_vector)
            # M_temp[i,:] = csdl.cross(thrust_origin[i,:]-ref_pt[0,:], F[i,:], axis=1)

        # self.print_var(thrust_origin)
            
        moments = csdl.cross(thrust_origin-ref_pt, F, axis=1)
        tempMoments = self.register_output('moments_BEM', moments)
        # self.print_var(tempMoments)

        tempSubtract = thrust_origin-ref_pt
        tempPrintSub = self.register_output('tempSubtract', tempSubtract)
        # self.print_var(tempPrintSub)
        # self.print_var(moments)
        M[:,0] = moments[:,0] * 0
        M[:,1] = moments[:,1]
        M[:,2] = moments[:,2] * 0
        
        # self.print_var(M)
        # self.print_var(F)
        #  # Post-Processing
        # T = self.declare_variable('T', shape=(num_nodes,))

        # ref_pt = self.declare_variable(name='ref_pt', shape=(num_nodes,3), units='m')
        # for vector_origin_pair_dict in BEM_pt_set_list:
        #     origin = vector_origin_pair_tuple['origin']
        #     vector = vector_origin_pair_tuple['vector']

        #     F = self.create_output(f'{vector.name}_F', shape=(num_nodes,3))
        #     M = self.create_output(f'{vector.name}_M', shape=(num_nodes,3))
        #
        #
        #
        #     # loop over pt set list

        #     for i in range(num_nodes):
        #         F[i,:] = csdl.expand(T[i],(1,3)) * n[i,:]
        #         M[i,0] = F[i,2] * (origin[i,1] - ref_pt[i,1])
        #         M[i,1] = F[i,2] * (origin[i,0] - ref_pt[i,0])
        #         M[i,2] = F[i,0] * (origin[i,1] - ref_pt[i,1])
