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
from ozone_models.ozone_actuation_outside import ODESystemModel
import python_csdl_backend
import numpy as np
from vedo import Points, Plotter
import cProfile

# number of timesteps
nt = 100    
dt = 0.01
h_vec = np.ones(nt-1) * dt    # A variable that needs to be created for Ozone

# Set initial conditions for all the states in the states dictionary
states = dict()
states['x']     = 0.0
states['y']     = 0.0
states['z']     = 0.0
states['u']     = 0.0
states['v']     = 0.0
states['w']     = 0.0
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
front_actuation_angle_end   = np.deg2rad(90)
front_actuation_array = np.linspace(front_actuation_angle_start, front_actuation_angle_end, nt)
front_act_dict['actuation_angle'] = front_actuation_array

# Acutation Angle Dict for Rear Wing
rear_act_dict = dict()
rear_actuation_angle_start = np.deg2rad(0.0)
rear_actuation_angle_end   = np.deg2rad(90)
rear_actuation_array = np.linspace(rear_actuation_angle_start, rear_actuation_angle_end, nt)
rear_act_dict['actuation_angle'] = rear_actuation_array

# Expand all the geometry pointsets to have a time-axis of size nt 
expanded_pointsets = dict()
for key,value in pointsets.items():
    if value.shape[0] == 1:
        temp = value
    else:
        temp = np.expand_dims(value, 0)
    expanded_pointsets[key] = np.repeat(temp, nt, 0)

# Defining the thrust vector dictionary for front wing 
front_thrust_vector_dict = dict()
front_thrust_vector_dict['front_left_nacelle1'] = (expanded_pointsets['front_left_nacelle1_origin'], expanded_pointsets['front_left_nacelle1_vector'])
front_thrust_vector_dict['front_left_nacelle2'] = (expanded_pointsets['front_left_nacelle2_origin'], expanded_pointsets['front_left_nacelle2_vector'])
front_thrust_vector_dict['front_left_nacelle3'] = (expanded_pointsets['front_left_nacelle3_origin'], expanded_pointsets['front_left_nacelle3_vector'])

# Defining the thrust vector dictionary for rear wing 
rear_thrust_vector_dict = dict()
rear_thrust_vector_dict['rear_left_nacelle1'] = (expanded_pointsets['rear_left_nacelle1_origin'], expanded_pointsets['rear_left_nacelle1_vector'])
rear_thrust_vector_dict['rear_left_nacelle2'] = (expanded_pointsets['rear_left_nacelle2_origin'], expanded_pointsets['rear_left_nacelle2_vector'])
rear_thrust_vector_dict['rear_left_nacelle3'] = (expanded_pointsets['rear_left_nacelle3_origin'], expanded_pointsets['rear_left_nacelle3_vector'])


# Defining an empty vlm dictionary        
front_vlm_mesh_dict = dict()
rear_vlm_mesh_dict = dict()

# Defining axis dictionary
front_axis_dict = dict()
front_axis_dict['front_wing_rotation_axis'] = expanded_pointsets['front_wing_rotation_axis']

rear_axis_dict = dict()
rear_axis_dict['rear_wing_rotation_axis'] = expanded_pointsets['rear_wing_rotation_axis']

axis_origin_front_wing_pt = expanded_pointsets['front_wing_axis_origin']
axis_origin_rear_wing_pt = expanded_pointsets['rear_wing_axis_origin']


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

# Define ODE system model
ode_problem = ODEProblem('RK4', 'time-marching', nt)
ode_problem.set_ode_system(ODESystemModel)
ode_problem.add_times(step_vector='h')

for key,value in states.items():
    ode_problem.add_state(key, f'd{key}_dt', initial_condition_name=f'{key}_0',
                          output=f'solved_{key}')

# Create main model that will contain actuation model and then call the ozone model
main_model = csdl.Model()
main_model.create_input('h', h_vec)

for key,value in states.items():
    main_model.create_input(f'{key}_0', val=value)

main_model.add(front_wing_actuation_model, 'front_wing_actuation_model')
main_model.add(rear_wing_actuation_model, 'rear_wing_actuation_model')


sim = python_csdl_backend.Simulator(front_wing_actuation_model, analytics=True)
sim.run(front_wing_actuation_model, 'front_wing_actuation_model')
# sim.check_partials(compact_print=True)
# sim.check_totals(compact_print=False)
# sim.visualize_implementation()

# PLOTTING THE VECTORS 
print(sim['front_left_nacelle1_vector_rotated'])

# Create the plot
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

# Plot the vector against time
plt.plot(sim['front_left_nacelle1_vector_rotated'][:,0], label='Vector')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Vector Value')
plt.title('Time History of a Vector')
plt.legend()

plt.grid(True)
plt.show()