import csdl
import matplotlib.pyplot as plt

from src.utils.data2csv import create_csv
from src.caddee.concept.geometry.geometry import Geometry

from ozone.api import ODEProblem, Wrap, NativeSystem

# from modopt.scipy_library import SLSQP
from modopt.snopt_library import SNOPT
from modopt.csdl_library import CSDLProblem

# FROM THE CADDEE REPO
from src.caddee.concept.geometry.geometry import Geometry
from src.caddee.concept.geometry.geocore.component import Component
from src.caddee.concept.geometry.geocore.utils.create_thrust_vector_origin_pair import generate_origin_vector_pair
from src.caddee.concept.geometry.geocore.utils.generate_corner_points import generate_corner_points
from src.caddee.concept.geometry.geocore.utils.generate_camber_mesh import generate_camber_mesh
from src.caddee.concept.geometry.geocore.utils.thrust_vector_creation import generate_thrust_vector

from geometry_files.both_wings_all_nacelles_tiltwing_geometry_points import both_wings_all_nacelles
from geometry_files.angle_axis_actuation import AngleAxisActuation
from ozone_models.ozone_actuation_outside_withVLM import ODESystemModel
from ozone_models.simple_forces_and_moments_model import simpleForcesAndMoments
import python_csdl_backend
import numpy as np
from vedo import Points, Plotter
import cProfile
from utils.plot3dfunction import plot_3d_array
from utils.plotMeshesAndVectors import plotMeshesAndVectors

# Conversions
ft2meters = 0.3048 

# number of timesteps
nt = 2
dt = 0.1
h_vec = np.ones(nt-1) * dt    # A variable that needs to be created for Ozone

# Set initial conditions for all the states in the states dictionary
states = dict()
states['x']     = 0.0
states['y']     = 0.0
states['z']     = -5.0
states['u']     = 50.0
states['v']     = 0.0
states['w']     = 0.00001
states['theta'] = 0.0
states['psi']   = 0.0
states['phi']   = 0.0
states['p']     = 0.0
states['q']     = 0.0
states['r']     = 0.0

# Call function to return all user defined pointsets
pointsets = both_wings_all_nacelles()

# Acutation Angle Dict for Front Wing
front_act_dict = dict()
front_actuation_angle_start = np.deg2rad(0.0)
front_actuation_angle_end   = np.deg2rad(0.0)
front_actuation_lower_bound = np.deg2rad(-5.0)
front_actuation_upper_bound = np.deg2rad(10.0)
front_actuation_scaler = 1e0
front_actuation_array = np.linspace(front_actuation_angle_start, front_actuation_angle_end, nt)
front_act_dict['front_actuation_angle_RAD'] = front_actuation_array

# Acutation Angle Dict for Rear Wing
rear_act_dict = dict()
rear_actuation_angle_start = np.deg2rad(0.0)
rear_actuation_angle_end   = np.deg2rad(0.0)
rear_actuation_lower_bound = np.deg2rad(-5.0)
rear_actuation_upper_bound = np.deg2rad(10.0)
rear_actuation_scaler = 1e0
rear_actuation_array = np.linspace(rear_actuation_angle_start, rear_actuation_angle_end, nt)
rear_act_dict['rear_actuation_angle_RAD'] = rear_actuation_array

# Reference Point [Meters]
# refPt = np.array([13.35494603, -4.123772014e-16 , 8.118931759]) * ft2meters
refPt = np.array([0, 0 , 0]) * ft2meters


# Thrust Dict for front wing [NEWTONS]
front_thrust_start        = 4000
front_thrust_end          = 4000
rear_thrust_start         = 4000
rear_thrust_end           = 4000
thrust_upper_bound  = 15000.0
thrust_lower_bound  = 0.0
thrust_scaler       = 1e-3

thrust = dict()
thrust['front_left_nacelle1_NEWTONS'] = np.linspace(front_thrust_start, front_thrust_end, nt)
thrust['front_left_nacelle2_NEWTONS'] = np.linspace(front_thrust_start, front_thrust_end, nt)
thrust['front_left_nacelle3_NEWTONS'] = np.linspace(front_thrust_start, front_thrust_end, nt)
thrust['rear_left_nacelle1_NEWTONS']  = np.linspace(rear_thrust_start, rear_thrust_end, nt)

# Expand all the geometry pointsets to have a time-axis of size nt 
expanded_pointsets = dict()
for key,value in pointsets.items():
    if value.shape[0] == 1:
        temp = value
    else:
        temp = np.expand_dims(value, 0)
    expanded_pointsets[key] = np.repeat(temp, nt, 0)

frontWingDict = dict()
frontWingDict['front_mesh_shape']  = (nt,9,3)
frontWingDict['front_wing_mesh']   = expanded_pointsets['front_left_wing_mesh_METERS_CG']
frontWingDict['fl1_thrust_origin'] = expanded_pointsets['front_left_nacelle1_origin_METERS_CG']
frontWingDict['fl2_thrust_origin'] = expanded_pointsets['front_left_nacelle2_origin_METERS_CG']
frontWingDict['fl3_thrust_origin'] = expanded_pointsets['front_left_nacelle3_origin_METERS_CG']
frontWingDict['fl1_thrust_vector'] = expanded_pointsets['front_left_nacelle1_vector_VSP']
frontWingDict['fl2_thrust_vector'] = expanded_pointsets['front_left_nacelle2_vector_VSP']
frontWingDict['fl3_thrust_vector'] = expanded_pointsets['front_left_nacelle3_vector_VSP']
frontWingDict['rot_axis_origin'] = expanded_pointsets['front_wing_axis_origin_METERS_CG']
frontWingDict['rot_axis_vector'] = expanded_pointsets['front_wing_rotation_axis']    

rearWingDict = dict()
rearWingDict['rear_mesh_shape']  = (nt,9,3)
rearWingDict['rear_wing_mesh']   = expanded_pointsets['rear_left_wing_mesh_METERS_CG']
rearWingDict['rl1_thrust_origin'] = expanded_pointsets['rear_left_nacelle1_origin_METERS_CG']
rearWingDict['rl1_thrust_vector'] = expanded_pointsets['rear_left_nacelle1_vector_VSP']
rearWingDict['rot_axis_origin'] = expanded_pointsets['rear_wing_axis_origin_METERS_CG']
rearWingDict['rot_axis_vector'] = expanded_pointsets['rear_wing_rotation_axis']    

# plotMeshesAndVectors(frontWingDict, rearWingDict, nt)

# plot_3d_array(expanded_pointsets['front_left_wing_mesh_METERS_CG'])


# Defining the thrust vector dictionary for front wing 
front_thrust_vector_dict = dict()
front_thrust_vector_dict['front_left_nacelle1'] = (expanded_pointsets['front_left_nacelle1_origin_METERS_CG'], expanded_pointsets['front_left_nacelle1_vector_VSP'])
front_thrust_vector_dict['front_left_nacelle2'] = (expanded_pointsets['front_left_nacelle2_origin_METERS_CG'], expanded_pointsets['front_left_nacelle2_vector_VSP'])
front_thrust_vector_dict['front_left_nacelle3'] = (expanded_pointsets['front_left_nacelle3_origin_METERS_CG'], expanded_pointsets['front_left_nacelle3_vector_VSP'])

# Defining the thrust vector dictionary for rear wing 
rear_thrust_vector_dict = dict()
rear_thrust_vector_dict['rear_left_nacelle1'] = (expanded_pointsets['rear_left_nacelle1_origin_METERS_CG'], expanded_pointsets['rear_left_nacelle1_vector_VSP'])

# Defining an empty vlm dictionary        
front_vlm_mesh_dict = dict()
front_vlm_mesh_dict['front_wing_vlm'] = expanded_pointsets['front_left_wing_mesh_METERS_CG']

rear_vlm_mesh_dict = dict()
rear_vlm_mesh_dict['rear_wing_vlm'] = expanded_pointsets['rear_left_wing_mesh_METERS_CG']

# Defining axis dictionary
front_axis_dict = dict()
front_axis_dict['front_wing_rotation_axis'] = expanded_pointsets['front_wing_rotation_axis']

rear_axis_dict = dict()
rear_axis_dict['rear_wing_rotation_axis'] = expanded_pointsets['rear_wing_rotation_axis']

axis_origin_front_wing_pt = expanded_pointsets['front_wing_axis_origin_METERS_CG']
axis_origin_rear_wing_pt = expanded_pointsets['rear_wing_axis_origin_METERS_CG']



# Defining the actuation model for front wing
front_wing_actuation_model = AngleAxisActuation(
    num_nodes = nt,
    thrust_vector_dict = front_thrust_vector_dict,
    vlm_mesh_dict = front_vlm_mesh_dict,
    axis_dict = front_axis_dict,
    actuation_angle_dict = front_act_dict,
    axis_origin_point = axis_origin_front_wing_pt,
)

# Defining the actuation model for rear wing
rear_wing_actuation_model = AngleAxisActuation(
    num_nodes = nt,
    thrust_vector_dict = rear_thrust_vector_dict,
    vlm_mesh_dict = rear_vlm_mesh_dict,
    axis_dict = rear_axis_dict,
    actuation_angle_dict = rear_act_dict,
    axis_origin_point = axis_origin_rear_wing_pt,
)

simpleForcesAndMomentsModel = simpleForcesAndMoments(
    nt=nt,
    front_thrust_vector_dict=front_thrust_vector_dict,
    rear_thrust_vector_dict=rear_thrust_vector_dict,
    thrust_dict=thrust,
    refPt=refPt
)


# Create main model that will contain actuation model and then call the ozone model
main_model = csdl.Model()
main_model.create_input('h', h_vec)
main_model.create_input('refPt', val=refPt)

for key,value in states.items():
    main_model.create_input(f'{key}_0', val=value)

for key,value in front_act_dict.items():
    front_act = main_model.create_input(key, val=value)
    main_model.add_design_variable(key, lower=front_actuation_lower_bound, upper=front_actuation_upper_bound,
                                        scaler=front_actuation_scaler)
    
    main_model.connect(key, f'{key}_mesh')
    
for key,value in rear_act_dict.items():
    rear_act = main_model.create_input(key, val=value)
    main_model.add_design_variable(key, lower=rear_actuation_lower_bound, upper=rear_actuation_upper_bound,
                                        scaler=rear_actuation_scaler)
    
    main_model.connect(key, f'{key}_mesh')

main_model.add(front_wing_actuation_model, 'front_wing_actuation_model')
main_model.add(rear_wing_actuation_model, 'rear_wing_actuation_model')

for key,value in thrust.items():
    if 'front_left_nacelle1_NEWTONS' in key:
        front_left_nacelle1_NEWTONS = main_model.create_input(key+'_thrust', val=value, shape=(nt,))
    elif 'front_left_nacelle2_NEWTONS' in key:
        front_left_nacelle2_NEWTONS = main_model.create_input(key+'_thrust', val=value, shape=(nt,))
    elif 'front_left_nacelle3_NEWTONS' in key:
        front_left_nacelle3_NEWTONS = main_model.create_input(key+'_thrust', val=value, shape=(nt,))
    elif 'rear_left_nacelle1_NEWTONS' in key:
        rear_left_nacelle1_NEWTONS = main_model.create_input(key+'_thrust', val=value, shape=(nt,))
        
    main_model.add_design_variable(key+'_thrust', lower=thrust_lower_bound, upper=thrust_upper_bound, scaler=thrust_scaler)

main_model.add(simpleForcesAndMomentsModel, 'simple_forces_moments_model')

# Define ODE system model
ode_problem = ODEProblem('RK4', 'time-marching', nt)
ode_problem.set_ode_system(ODESystemModel)
ode_problem.add_times(step_vector='h')

ode_problem.add_parameter('front_wing_vlm_rotated', dynamic=True, shape=(nt,3,3,3))
ode_problem.add_parameter('rear_wing_vlm_rotated',  dynamic=True, shape=(nt,3,3,3))

ode_problem.add_parameter('forces_moments_Fx', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('forces_moments_Fy', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('forces_moments_Fz', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('forces_moments_Mx', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('forces_moments_My', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('forces_moments_Mz', dynamic=True, shape=(nt,1))
ode_problem.add_parameter('refPt', dynamic=False, shape=(3,))

for key,value in states.items():
    ode_problem.add_state(key, f'd{key}_dt', initial_condition_name=f'{key}_0',
                          output=f'solved_{key}')
    
ode_problem.add_profile_output('total_Fx', shape=(1,))
ode_problem.add_profile_output('total_Fy', shape=(1,))
ode_problem.add_profile_output('total_Fz', shape=(1,))

ode_problem.add_profile_output('total_Mx', shape=(1,))
ode_problem.add_profile_output('total_My', shape=(1,))
ode_problem.add_profile_output('total_Mz', shape=(1,))

ode_problem.add_profile_output('du_dt', shape=(1,))
ode_problem.add_profile_output('dw_dt', shape=(1,))
ode_problem.add_profile_output('dq_dt', shape=(1,))


ode_problem.set_profile_system(ODESystemModel)
main_model.add(ode_problem.create_solver_model(), 'subgroup')

main_model.declare_variable('total_Fx', shape=(nt,1))
main_model.declare_variable('total_Fy', shape=(nt,1))
main_model.declare_variable('total_Fz', shape=(nt,1))
main_model.declare_variable('total_Mx', shape=(nt,1))
main_model.declare_variable('total_My', shape=(nt,1))
main_model.declare_variable('total_Mz', shape=(nt,1))

du_dt = main_model.declare_variable('du_dt', shape=(nt,1))
dw_dt = main_model.declare_variable('dw_dt', shape=(nt,1))
dq_dt = main_model.declare_variable('dq_dt', shape=(nt,1))

x = main_model.declare_variable('solved_x', shape=(nt,1))
z = main_model.declare_variable('solved_z', shape=(nt,1))
u = main_model.declare_variable('solved_u', shape=(nt,1))
w = main_model.declare_variable('solved_w', shape=(nt,1))
theta = main_model.declare_variable('solved_theta', shape=(nt,1))
q = main_model.declare_variable('solved_q', shape=(nt,1))

flipZ = main_model.create_input('flipZ', val=-1*np.ones((nt,1)), shape=(nt,1))
vertVel = main_model.create_output(name='vertVel', shape=(nt,1)) 
vertVel[:, 0] = w * flipZ   # Vertical Velocity

tol = main_model.create_input('tol', val=np.ones((nt,1))*1e-8)
thetaTol = theta + tol
thetaConstraint = csdl.max( csdl.pnorm(thetaTol, axis=1) * 100 ) / 100
main_model.register_output('thetaConstraint', thetaConstraint)


########################################################
# Computing acceleration using back-difference of rates
########################################################
# acceleration in x
ddx_1 = main_model.create_output('accel_x_1', shape=(nt-1,1))
ddx_2 = main_model.create_output('accel_x_2', shape=(nt-1,1))

ddx_2[0:nt-1,0] = u[1:nt,0]
ddx_1[0:nt-1,0] = u[0:nt-1,0]

ddx = main_model.create_output('accel_x', shape=(nt-1,1))
ddx[0:nt-1,0] = ddx_2 - ddx_1

# acceleration in z
ddz_1 = main_model.create_output('accel_z_1', shape=(nt-1,1))
ddz_2 = main_model.create_output('accel_z_2', shape=(nt-1,1))

ddz_2[0:nt-1,0] = w[1:nt,0]
ddz_1[0:nt-1,0] = w[0:nt-1,0]

ddz = main_model.create_output('accel_z', shape=(nt-1,1))
ddz[0:nt-1,0] = ddz_2 - ddz_1

# acceleration in theta
ddtheta_1 = main_model.create_output('accel_theta_1', shape=(nt-1,1))
ddtheta_2 = main_model.create_output('accel_theta_2', shape=(nt-1,1))

ddtheta_2[0:nt-1,0] = q[1:nt,0]
ddtheta_1[0:nt-1,0] = q[0:nt-1,0]

ddtheta = main_model.create_output('accel_theta', shape=(nt-1,1))
ddtheta[0:nt-1,0] = ddtheta_2 - ddtheta_1

# Constraint for minimum altitude possible
alt = z * np.ones((nt,1))*-1
zdesire_final = main_model.create_input('final_alt', val=np.ones((1,1))*5.0)
finalPositionConstraint = alt[nt-1,0] - zdesire_final
main_model.register_output('finalPositionConstraint', finalPositionConstraint)

# Constraint on Qdot 
finalQdot = main_model.create_output('finalQdot', shape=(1,1))
finalQdot[0,0] = dq_dt[nt-1,0]

# Constraint on Q
finalQ = main_model.create_output('finalQ', shape=(1,1))
finalQ[0,0] = q[nt-1,0]

# Constraint on Udot
finalUdot = main_model.create_output('finalUdot', shape=(1,1))
finalUdot[0,0] = du_dt[nt-1,0]

# Constraint on U
finalU = main_model.create_output('finalU', shape=(1,1))
finalU[0,0] = u[nt-1,0]

# Constraint on Wdot
finalWdot = main_model.create_output('finalWdot', shape=(1,1))
finalWdot[0,0] = dw_dt[nt-1,0]

# Constraint on W
finalW = main_model.create_output('finalW', shape=(1,1))
finalW[0,0] = w[nt-1,0]

altConstraint = main_model.create_output(name='altConstraint', shape=(1, ))
altConstraint[0] = csdl.min(alt)

# # Developing rotation rate constraints for the wings
# rearWingRotRate1 = main_model.create_output('RearWingRotRate1', shape=(nt-1,1))
# rearWingRotRate1[0:nt-1] = rear_act[0:nt-1]
# rearWingRotRate2 = main_model.create_output('RearWingRotRate2', shape=(nt-1,1))
# rearWingRotRate2[0:nt-1] = rear_act[1:nt]
# rearWingRotRate = (rearWingRotRate2 - rearWingRotRate1)
# main_model.register_output('rearWingRotRate', rearWingRotRate)
# rearWingRotRateMax = csdl.max( (csdl.pnorm(rearWingRotRate, axis=1) )) 
# main_model.register_output('rearMaxRotRate', rearWingRotRateMax)

# frontWingRotRate1 = main_model.create_output('FrontWingRotRate1', shape=(nt-1,1))
# frontWingRotRate1[0:nt-1] = front_act[0:nt-1]
# frontWingRotRate2 = main_model.create_output('FrontWingRotRate2', shape=(nt-1,1))
# frontWingRotRate2[0:nt-1] = front_act[1:nt]
# frontWingRotRate = (frontWingRotRate2 - frontWingRotRate1)
# main_model.register_output('frontWingRotRate', frontWingRotRate)
# frontWingRotRateMax = csdl.max( (csdl.pnorm(frontWingRotRate, axis=1) ) ) 
# main_model.register_output('frontMaxRotRate', frontWingRotRateMax)

###########################################
#            CONSTRAINTS FUNCTIONS        #          
###########################################
# thetaConstraintLower = -0.1745
# thetaConstraintUpper = 0.1745
thetaConstraintLower = -0.1
thetaConstraintUpper = 0.1
thetaConstraintScaler= 1e-1

altConstraintLower = 0

frontMaxRotRateLower = -0.02
frontMaxRotRateUpper = 0.02
frontMaxRotRateScaler = 1e0

rearMaxRotRateLower = -0.02
rearMaxRotRateUpper = 0.02
rearMaxRotRateScaler = 1e0

finalPositionLower = -0.1
finalPositionUpper = 0.1

finalQdotUpper = 1e-4
finalQUpper = 1e-4
finalUdotUpper  = 1e-4
finalUUpper   = 1e-4 
finalWdotUpper = 1e-4
finalWUpper = 1e-4

# main_model.add_constraint('thetaConstraint', lower=thetaConstraintLower, upper=thetaConstraintUpper, scaler=thetaConstraintScaler )
# main_model.add_constraint('altConstraint',  lower=altConstraintLower)
# main_model.add_constraint('frontMaxRotRate',lower=frontMaxRotRateLower, upper=frontMaxRotRateUpper, scaler=frontMaxRotRateScaler)
# main_model.add_constraint('rearMaxRotRate', lower=rearMaxRotRateLower, upper=rearMaxRotRateUpper, scaler=rearMaxRotRateScaler)
# main_model.add_constraint('finalPositionConstraint', lower=finalPositionLower, upper=finalPositionUpper)

# main_model.add_constraint('finalQdot', upper=finalQdotUpper)
# main_model.add_constraint('finalQ',    upper=finalQUpper)
# main_model.add_constraint('finalUdot', upper=finalUdotUpper)
# main_model.add_constraint('finalU',    upper=finalUUpper)
# main_model.add_constraint('finalWdot', upper=finalWUpper)
# main_model.add_constraint('finalW',    upper=finalWUpper)

###########################################
#            OBJECTIVE FUNCTION           #          
###########################################
# Minimum effort objective function
obj_vec      = main_model.create_output('obj_vec', shape=(1,4))
# obj_vec[:,0] = ( csdl.reshape(front_act, new_shape=(nt,1)) * csdl.reshape(front_act, new_shape=(nt,1)) ) / (front_actuation_upper_bound*front_actuation_upper_bound*nt)
# obj_vec[:,1] = ( csdl.reshape(rear_act, new_shape=(nt,1))  * csdl.reshape(rear_act, new_shape=(nt,1))  ) / (rear_actuation_upper_bound*rear_actuation_upper_bound*nt)
# obj_vec[:,2] = ( csdl.reshape(front_left_nacelle1_NEWTONS, new_shape=(nt,1)) * csdl.reshape(front_left_nacelle1_NEWTONS, new_shape=(nt,1)) ) / (thrust_upper_bound*thrust_upper_bound*nt)
# obj_vec[:,3] = ( csdl.reshape(front_left_nacelle2_NEWTONS, new_shape=(nt,1)) * csdl.reshape(front_left_nacelle2_NEWTONS, new_shape=(nt,1)) ) / (thrust_upper_bound*thrust_upper_bound*nt)
# obj_vec[:,4] = ( csdl.reshape(front_left_nacelle3_NEWTONS, new_shape=(nt,1)) * csdl.reshape(front_left_nacelle3_NEWTONS, new_shape=(nt,1)) ) / (thrust_upper_bound*thrust_upper_bound*nt)
# obj_vec[:,5] = ( csdl.reshape(rear_left_nacelle1_NEWTONS, new_shape=(nt,1)) * csdl.reshape(rear_left_nacelle1_NEWTONS, new_shape=(nt,1)) ) / (thrust_upper_bound*thrust_upper_bound*nt)

obj_vec[:,0] = ( csdl.reshape(finalQdot, new_shape=(1,1)) * csdl.reshape(finalQdot, new_shape=(1,1)) )
obj_vec[:,1] = ( csdl.reshape(finalQ, new_shape=(1,1)) * csdl.reshape(finalQ, new_shape=(1,1))  ) 
obj_vec[:,2] = ( csdl.reshape(finalUdot, new_shape=(1,1)) * csdl.reshape(finalUdot, new_shape=(1,1)) )
# obj_vec[:,3] = ( csdl.reshape(finalU, new_shape=(1,1)) * csdl.reshape(finalU, new_shape=(1,1)) ) 
obj_vec[:,3] = ( csdl.reshape(finalWdot, new_shape=(1,1)) * csdl.reshape(finalWdot, new_shape=(1,1)) ) 
# obj_vec[:,5] = ( csdl.reshape(finalW, new_shape=(1,1)) * csdl.reshape(finalW, new_shape=(1,1)) )  


squaredFinalPos = csdl.reshape(finalPositionConstraint, new_shape=(1,)) * csdl.reshape(finalPositionConstraint, new_shape=(1,)) * 1e-4
obj_sum  = (csdl.sum(obj_vec) ) 
main_model.register_output('objective', obj_sum)
main_model.print_var(obj_sum)

main_model.add_objective('objective')

sim = python_csdl_backend.Simulator(main_model, analytics=True)
sim.run()

major_iterations = 100000
major_optimality = 1e-8
major_feasibility = 1e-5
linesearch_tolerance = 0.99
major_step_size = 0.1
superbasics_limit = 1000
prob = CSDLProblem(problem_name='Equalsplusminus5constraint', simulator=sim)
# optimizer = SNOPT(
#     prob, 
#     Major_iterations = major_iterations,
#     Major_optimality= major_optimality, 
#     Major_feasibility= major_feasibility,
#     Superbasics_limit=superbasics_limit,
#     Linesearch_tolerance=linesearch_tolerance,
#     Major_step_limit=major_step_size,
#     append2file=True
# )
# optimizer.solve()
# optimizer.print_results()


# Plots for Confirmation of rotations
frontWingDict = dict()
frontWingDict['front_mesh_shape']   = (nt,3,3,3)
frontWingDict['front_wing_mesh']    = sim['front_wing_vlm_rotated']
frontWingDict['fl1_thrust_origin']  = sim['front_left_nacelle1_origin_rotated_METERS_NED_CG']
frontWingDict['fl2_thrust_origin']  = sim['front_left_nacelle2_origin_rotated_METERS_NED_CG']
frontWingDict['fl3_thrust_origin']  = sim['front_left_nacelle3_origin_rotated_METERS_NED_CG']
frontWingDict['fl1_thrust_vector']  = sim['front_left_nacelle1_vector_rotated_NED']
frontWingDict['fl2_thrust_vector']  = sim['front_left_nacelle2_vector_rotated_NED']
frontWingDict['fl3_thrust_vector']  = sim['front_left_nacelle3_vector_rotated_NED']
frontWingDict['rot_axis_origin']    = sim['front_left_nacelle1_axis_origin_pt_METERS_VSP']
frontWingDict['rot_axis_vector']    = sim['front_wing_rotation_axis_axis_vector']
rearWingDict = dict()
rearWingDict['rear_mesh_shape']   = (nt,3,3,3)
rearWingDict['rear_wing_mesh']    = sim['rear_wing_vlm_rotated']
rearWingDict['rl1_thrust_origin'] = sim['rear_left_nacelle1_origin_rotated_METERS_NED_CG']
rearWingDict['rl1_thrust_vector'] = sim['rear_left_nacelle1_vector_rotated_NED']
rearWingDict['rot_axis_origin']   = sim['rear_left_nacelle1_axis_origin_pt_METERS_VSP']
rearWingDict['rot_axis_vector']   = sim['rear_wing_rotation_axis_axis_vector']
plotMeshesAndVectors(frontWingDict, rearWingDict, nt)


dirpath = '/home/alexander/Documents/OptResults/HoverTrimOptimization_singleConstraintV2'
'''  
Save Input Data to CSV
'''
# filename = dirpath +'/INPUTS.csv'
# inputDict = dict()
# inputDict['nt'] = np.array([nt])
# inputDict['dt'] = np.array([dt])
# inputDict['major_iterations'] = np.array([major_iterations])
# inputDict['major_optimality'] = np.array([major_optimality])
# inputDict['major_feasibility'] = np.array([major_feasibility])
# inputDict['linesearch_tolerance'] = np.array([linesearch_tolerance])
# inputDict['superbasics_limit'] = np.array([superbasics_limit])
# inputDict['major_step_size'] = np.array([major_step_size])
# inputDict['refPt'] = refPt
# inputDict['IC_x'] = np.array([states['x']])
# inputDict['IC_y'] = np.array([states['y']])
# inputDict['IC_z'] = np.array([states['z']])
# inputDict['IC_u'] = np.array([states['u']])
# inputDict['IC_v'] = np.array([states['v']])
# inputDict['IC_w'] = np.array([states['w']])
# inputDict['IC_theta'] = np.array([states['theta']])
# inputDict['IC_psi'] = np.array([states['psi']])
# inputDict['IC_phi'] = np.array([states['phi']])
# inputDict['IC_p'] = np.array([states['p']])
# inputDict['IC_q'] = np.array([states['q']])
# inputDict['IC_r'] = np.array([states['r']])
# inputDict['front_thrust_start'] = np.array([front_thrust_start])
# inputDict['front_thrust_stop']  = np.array([front_thrust_end])
# inputDict['rear_thrust_start'] = np.array([rear_thrust_start])
# inputDict['rear_thrust_stop']  = np.array([rear_thrust_end])
# inputDict['thrust_upper_bound'] = np.array([thrust_lower_bound])
# inputDict['thrust_lower_bound']  = np.array([thrust_upper_bound])
# inputDict['thrust_scaler'] = np.array([thrust_scaler])
# inputDict['front_act_start']  = np.array([front_actuation_angle_start])
# inputDict['front_act_stop'] = np.array([front_actuation_angle_end])
# inputDict['front_act_scaler'] = np.array([front_actuation_scaler])
# inputDict['rear_act_start']  = np.array([rear_actuation_angle_start])
# inputDict['rear_act_stop'] = np.array([rear_actuation_angle_end])
# inputDict['objective'] = sim['objective']
# create_csv(filename, inputDict)

'''  
Save DV Data to CSV
'''
# filename = dirpath +'/DV.csv'
# dvDict = dict()
# dvDict['front_thrust_start'] = np.array([front_thrust_start])
# dvDict['front_thrust_stop']  = np.array([front_thrust_end])
# dvDict['rear_thrust_start'] = np.array([rear_thrust_start])
# dvDict['rear_thrust_stop']  = np.array([rear_thrust_end])
# dvDict['thrust_upper_bound'] = np.array([thrust_lower_bound])
# dvDict['thrust_lower_bound']  = np.array([thrust_upper_bound])
# dvDict['thrust_scaler'] = np.array([thrust_scaler])
# dvDict['front_act_start']  = np.array([front_actuation_angle_start])
# dvDict['front_act_stop'] = np.array([front_actuation_angle_end])
# dvDict['front_act_scaler'] = np.array([front_actuation_scaler])
# dvDict['rear_act_start']  = np.array([rear_actuation_angle_start])
# dvDict['rear_act_stop'] = np.array([rear_actuation_angle_end])
# dvDict['rear_act_scaler'] = np.array([rear_actuation_scaler])
# dvDict['rear_act_scaler'] = np.array([rear_actuation_scaler])
# create_csv(filename, dvDict)

'''  
Save CONSTRAINTS Data to CSV
'''
# filename = dirpath +'/CONSTRAINTS.csv'
# constraintDict = dict()
# constraintDict['thetaConstraintLower'] = np.array([thetaConstraintLower])
# constraintDict['thetaConstraintUpper']  = np.array([thetaConstraintUpper])
# constraintDict['thetaConstraintScaler'] = np.array([thetaConstraintScaler])
# constraintDict['altConstraintLower']  = np.array([altConstraintLower])
# constraintDict['frontMaxRotRateLower'] = np.array([frontMaxRotRateLower])
# constraintDict['frontMaxRotRateUpper']  = np.array([frontMaxRotRateUpper])
# constraintDict['frontMaxRotRateScaler'] = np.array([frontMaxRotRateScaler])
# constraintDict['finalPositionLower'] = np.array([finalPositionLower])
# constraintDict['finalPositionUpper']  = np.array([finalPositionUpper])
# constraintDict['finalQdotUpper'] = np.array([finalQdotUpper])
# constraintDict['finalQUpper'] = np.array([finalQUpper])
# constraintDict['finalQUpper'] = np.array([finalQUpper])
# constraintDict['finalUdotUpper'] = np.array([finalUdotUpper])
# constraintDict['finalUUpper'] = np.array([finalUUpper])
# constraintDict['finalWdotUpper'] = np.array([finalWdotUpper])
# constraintDict['finalWUpper'] = np.array([finalWUpper])
# create_csv(filename, constraintDict)

'''  
Save Output Data to CSV
'''
# filename = dirpath +'/OUTPUTS.csv'
# outputDict = dict()
# outputDict['total_Fx'] = sim['total_Fx']
# outputDict['total_Fy'] = sim['total_Fy']
# outputDict['total_Fz'] = sim['total_Fz']
# outputDict['total_Mx'] = sim['total_Mx']
# outputDict['total_My'] = sim['total_My']
# outputDict['total_Mz'] = sim['total_Mz']
# outputDict['solved_x'] = sim['solved_x']
# outputDict['solved_u'] = sim['solved_u']
# outputDict['du_dt'] = sim['du_dt']
# outputDict['solved_z'] = sim['solved_z']
# outputDict['solved_w'] = sim['solved_w']
# outputDict['dw_dt'] = sim['dw_dt']
# outputDict['solved_theta'] = sim['solved_theta']
# outputDict['solved_q'] = sim['solved_q']
# outputDict['dq_dt'] = sim['dq_dt']
# outputDict['front_actuation_angle_RAD'] = sim['front_actuation_angle_RAD']
# outputDict['rear_actuation_angle_RAD'] = sim['rear_actuation_angle_RAD']
# outputDict['front_left_nacelle1_NEWTONS_thrust'] = sim['front_left_nacelle1_NEWTONS_thrust']
# outputDict['front_left_nacelle2_NEWTONS_thrust'] = sim['front_left_nacelle2_NEWTONS_thrust']
# outputDict['front_left_nacelle3_NEWTONS_thrust'] = sim['front_left_nacelle3_NEWTONS_thrust']
# outputDict['rear_left_nacelle1_NEWTONS_thrust'] = sim['rear_left_nacelle1_NEWTONS_thrust']
# create_csv(filename, outputDict)


# sim.check_partials(compact_print=True)
# sim.check_totals(compact_print=True)
# sim.visualize_implementation()

# PLOTTING THE VECTORS 
# print('----------- VSP VECTORS ----------------')
# print(sim['front_left_nacelle1_vector_rotated_VSP'])
# print(sim['front_left_nacelle2_vector_rotated_VSP'])
# print(sim['front_left_nacelle3_vector_rotated_VSP'])
# print(sim['rear_left_nacelle1_vector_rotated_VSP'])

# print('----------- NED VECTORS ----------------')
# print(sim['front_left_nacelle1_vector_rotated_NED'])
# print(sim['front_left_nacelle2_vector_rotated_NED'])
# print(sim['front_left_nacelle3_vector_rotated_NED'])
# print(sim['rear_left_nacelle1_vector_rotated_NED'])

# print('----------- NED CG ORIGIN ----------------')
print(sim['front_wing_vlm_rotated'])
print(expanded_pointsets['front_left_wing_mesh_METERS_NED_CG'])
# print(sim['front_left_nacelle3_origin_rotated_METERS_NED_CG'])
# print(sim['rear_left_nacelle1_origin_rotated_METERS_NED_CG'])

# print('------ FORCES AND MOMENTS -------')
# print(sim['forces_moments_Fx'])
# print(sim['forces_moments_Fy'])
# print(sim['forces_moments_Fz'])

# print(sim['forces_moments_Mx'])
# print(sim['forces_moments_My'])
# print(sim['forces_moments_Mz'])

print('------ FORCES AND MOMENTS -------')
print(sim['total_Fx'])
print(sim['total_Fy'])
print(sim['total_Fz'])

print(sim['total_Mx'])
print(sim['total_My'])
print(sim['total_Mz'])

print('---------- Thrust --------------')
for key,value in thrust.items():
    print(f'{key}_thrust', sim[f'{key}_thrust'])

print('---------- Actuation Angles --------------')
print('Front Act:', sim['front_actuation_angle_RAD']*180.0/np.pi)
print('Rear Act:', sim['rear_actuation_angle_RAD']*180.0/np.pi)

print('---------- STATES --------------')
print('X:', sim['solved_x'])
print('U:',sim['solved_u'])
print('Udot:', sim['du_dt'])
print('Z:',sim['solved_z'])
print('W:',sim['solved_w'])
print('Wdot:', sim['dw_dt'])
print('Theta:',sim['solved_theta']*180.0/np.pi)
print('Q:',sim['solved_q']*180.0/np.pi)
print('Qdot:',sim['dq_dt']*180.0/np.pi)


# print(sim['front_left_nacelle1_thrust_vector_mult'])
# print(sim['front_left_nacelle2_thrust_vector_mult'])
# print(sim['front_left_nacelle2_thrust_vector_mult'])
# print(sim['rear_left_nacelle1_thrust_vector_mult'])


# Create the plot
# plt.figure(figsize=(10, 6))  # Optional: Set the figure size

# # Plot the vector against time
# plt.plot(sim['front_left_nacelle1_vector_rotated'][:,0], label='Vector')

# # Add labels and title
# plt.xlabel('Time')
# plt.ylabel('Vector Value')
# plt.title('Time History of a Vector')
# plt.legend()

# plt.grid(True)
# plt.show()