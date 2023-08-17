import csdl
import matplotlib.pyplot as plt

from src.utils.data2csv import create_csv
from src.caddee.concept.geometry.geometry import Geometry

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

import python_csdl_backend
import numpy as np
from vedo import Points, Plotter
import cProfile

# number of timesteps
nt = 100     

pointsets = both_wings_all_nacelles()

# Acutation Angle Dict

act_dict = dict()
actuation_angle_start = 0
actuation_angle_end   = 90
actuation_array = np.linspace(actuation_angle_start, actuation_angle_end, nt)
act_dict['actuation_angle'] = actuation_array

# Expand all the geometry pointsets to have a time-axis of size nt 
expanded_pointsets = dict()
for key,value in pointsets.items():
    if value.shape[0] == 1:
        temp = value
    else:
        temp = np.expand_dims(value, 0)
    expanded_pointsets[key] = np.repeat(temp, nt, 0)

# Defining the thrust vector dictionary
thrust_vector_dict = dict()
thrust_vector_dict['front_left_nacelle1'] = (expanded_pointsets['front_left_nacelle1_origin'], expanded_pointsets['front_left_nacelle1_vector'])
# thrust_vector_dict['front_left_nacelle1_vector'] = expanded_pointsets['front_left_nacelle1_vector']
thrust_vector_dict['front_left_nacelle2'] = (expanded_pointsets['front_left_nacelle2_origin'], expanded_pointsets['front_left_nacelle2_vector'])
# thrust_vector_dict['front_left_nacelle2_vector'] = expanded_pointsets['front_left_nacelle2_vector']
thrust_vector_dict['front_left_nacelle3'] = (expanded_pointsets['front_left_nacelle3_origin'], expanded_pointsets['front_left_nacelle3_vector'])
# thrust_vector_dict['front_left_nacelle3_vector'] = expanded_pointsets['front_left_nacelle3_vector']

# Defining an empty vlm dictionary        
vlm_mesh_dict = dict()

# Defining axis dictionary
axis_dict = dict()
axis_dict['front_wing_rotation_axis'] = expanded_pointsets['front_wing_rotation_axis']

axis_origin_front_wing_pt = expanded_pointsets['front_wing_axis_origin']


actuation_model = AngleAxisActuation(
    num_nodes = nt,
    thrust_vector_dict = thrust_vector_dict,
    vlm_mesh_dict = vlm_mesh_dict,
    axis_dict = axis_dict,
    actuation_angle_dict = act_dict,
    axis_origin_point = axis_origin_front_wing_pt,
)
sim = python_csdl_backend.Simulator(actuation_model, analytics=True)
sim.run()
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