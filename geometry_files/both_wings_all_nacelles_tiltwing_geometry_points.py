import numpy as np
from src.caddee.concept.geometry.geometry import Geometry
from src.caddee.concept.geometry.geocore.component import Component

from src.caddee.concept.geometry.geocore.utils.create_thrust_vector_origin_pair import generate_origin_vector_pair
from src.caddee.concept.geometry.geocore.utils.generate_corner_points import generate_corner_points
from src.caddee.concept.geometry.geocore.utils.generate_camber_mesh import generate_camber_mesh
from src.caddee.concept.geometry.geocore.utils.thrust_vector_creation import generate_thrust_vector
from vedo import Points, Plotter, LegendBox

from utils.plotMeshesAndVectors import plotMeshesAndVectors
from src import STP_FILES_FOLDER


def both_wings_all_nacelles():
    mm2ft = 304.8
    ft2meters = 0.3048 

    up_direction = np.array([0., 0., 1.])
    down_direction = np.array([0., 0., -1.])

    stp_path = STP_FILES_FOLDER / 'both_wings_all_nacelles.stp'


    top_surfaces = ['FrontWing, 0, 4', 'FrontWing, 0, 8', 'FrontWing, 0, 12']
        #'FrontWing, 1, 22', 'FrontWing, 1, 26', 'FrontWing, 1, 30'
        # Top Surfaces of the Front Wing

    bot_surfaces = ['FrontWing, 0, 3', 'FrontWing, 0, 7', 'FrontWing, 0, 11']
    #     'FrontWing, 1, 21', 'FrontWing, 1, 25', 'FrontWing, 1, 29'
    # ]  # Bot Surfaces of the Front Wing

    fsurfaces = top_surfaces + bot_surfaces

    top_surfaces_r = ['RearWing, 0, 88'] 
                    #'RearWing, 1, 98']

    bot_surfaces_r = ['RearWing, 0, 87']  
                    #'RearWing, 1, 97']

    rsurfaces = top_surfaces_r + bot_surfaces_r

    geo = Geometry()
    geo.read_file(file_name=stp_path)

    # DEFINE FRONT AND REAR WING COMPONENTS
    front_wing = Component(
        stp_entity_names=['FrontWing'],
        name='front_wing')  # Creating a wing component and naming it wing
    geo.add_component(front_wing)

    rear_wing = Component(
        stp_entity_names=rsurfaces,
        name='rear_wing')  # Creating a wing component and naming it wing
    geo.add_component(rear_wing)

    # CREATE FRONT WING LEFT NACELLE COMPONENTS
    flnacelle1         = Component(stp_entity_names=['FrontLeftNacelle1'], name='flnacelle1')
    geo.add_component(flnacelle1)

    flnacelle2          = Component(stp_entity_names=['FrontLeftNacelle2'], name='flnacelle2')
    geo.add_component(flnacelle2)

    flnacelle3          = Component(stp_entity_names=['FrontLeftNacelle3'], name='flnacelle3')
    geo.add_component(flnacelle3)

    # CREATE REAR WING LEFT NACELLE COMPONENT
    rlnacelle1          = Component(stp_entity_names=['RearLeftNacelle1'], name='rlnacelle1')
    geo.add_component(rlnacelle1)

    # CREATING 2 FRONT WING CAMBER SURFACE MESHES
    point00 = np.array([3962.33, 6662.76, 2591]) / mm2ft  # Left side of fwing
    point01 = np.array([4673.88, 6662.76, 2590.8]) / mm2ft  # Left side of fwing
    point10 = np.array([2711.276, 0.0, 2591]) / mm2ft  # Mid of fwing
    point11 = np.array([3727.767, 0.0, 2591]) / mm2ft  # Mid of fwing

    fwing_lead_left, fwing_trail_left, fwing_lead_left_mid, fwing_trail_left_mid = generate_corner_points(
                                                geo, "front_wing", point00, point01, point10, point11)
    fwing_left_top, fwing_left_bot, fwing_chord_surface_left, fwing_camber_surface_left = generate_camber_mesh(
                                                geo, fwing_lead_left, fwing_trail_left, fwing_lead_left_mid,
                                                fwing_trail_left_mid, top_surfaces, bot_surfaces, "front_wing_left")

    # CREATING 2 REAR WING CAMBER SURFACE MESHES
    point00 = np.array([8277.012, 1536.2898, 4006.5]) / mm2ft  # Left side of fwing
    point01 = np.array([9123.5639, 1536.2898, 4006.5]) / mm2ft  # Left side of fwing
    point10 = np.array([8065.374, 0.0, 4006.5]) / mm2ft  # Mid of fwing
    point11 = np.array([9123.563, 0.0, 4006.5]) / mm2ft  # Mid of fwing

    rwing_lead_left, rwing_trail_left, rwing_lead_left_mid, rwing_trail_left_mid = generate_corner_points(
        geo, "rear_wing", point00, point01, point10, point11)

    rwing_left_top, rwing_left_bot, rwing_chord_surface_left, rwing_camber_surface_left = generate_camber_mesh(
        geo, rwing_lead_left, rwing_trail_left, rwing_lead_left_mid,
        rwing_trail_left_mid, top_surfaces_r, bot_surfaces_r, "rear_wing_left")

    # COORDINATES FOR FRONT WING  LEFT NACELLES
    frontLeftWingNacelle1TopPoint = np.array([2537.68581,	2258.67681,	2672.9])/ mm2ft
    frontLeftWingNacelle1BotPoint = np.array([2537.68581,	2258.67681,	2288.0])/ mm2ft

    flnacelle1_thrust=generate_origin_vector_pair(geo, frontLeftWingNacelle1TopPoint, frontLeftWingNacelle1BotPoint, flnacelle1)

    frontLeftWingNacelle2TopPoint = np.array([2952.06105,	4457.3894,	2672.69567])/ mm2ft
    frontLeftWingNacelle2BotPoint = np.array([2952.06105,	4457.3894,	2288.647675])/ mm2ft

    flnacelle2_thrust=generate_origin_vector_pair(geo, frontLeftWingNacelle2TopPoint, frontLeftWingNacelle2BotPoint, flnacelle2)

    frontLeftWingNacelle3TopPoint = np.array([3367.3566,	6662.762798,	2782.825])/ mm2ft
    frontLeftWingNacelle3BotPoint = np.array([3367.3566,	6662.762798,	2398.774])/ mm2ft

    flnacelle3_thrust=generate_origin_vector_pair(geo, frontLeftWingNacelle3TopPoint, frontLeftWingNacelle3BotPoint, flnacelle3)

    # COORDINATES FOR REAR WING LEFT NACELLES
    rearLeftWingNacelle1TopPoint = np.array([7679.89761,	1536.2898,	4197.9729])/ mm2ft
    rearLeftWingNacelle1BotPoint = np.array([7679.89761,	1536.2898,	3813.9249])/ mm2ft
    rlnacelle1_thrust=generate_origin_vector_pair(geo, rearLeftWingNacelle1TopPoint, rearLeftWingNacelle1BotPoint, rlnacelle1)

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # DEFINE THE POINTS TO CREATE AN AXIS OF ROTATION FOR FRONT AND REAR WINGS
    # LEFT MIDDLE WING POINTS
    front_middle_wing = np.array([2711.2761624, 0, 2591.0])/mm2ft
    rear_middle_wing  = np.array([3727.7673984, 0.0, 2591.0])/mm2ft

    frontLeftMiddlePt, frontLeftMiddlePtCoord = geo.project_points(front_middle_wing, projection_direction = down_direction, projection_targets_names=["front_wing"],plot=False)
    rearLeftMiddlePt, rearLeftMiddlePtCoord = geo.project_points(rear_middle_wing, projection_direction = down_direction, projection_targets_names=["front_wing"], plot=False)

    # To line up the x-coordinates with the FrontLeft3 nacelle 
    temp  = ((flnacelle3_thrust[0].physical_coordinates[0][0] - rearLeftMiddlePtCoord[0][0]) / (frontLeftMiddlePtCoord[0][0] - rearLeftMiddlePtCoord[0][0]))
    alpha = 1 - temp 
    middleLeftIntrp = geo.perform_linear_interpolation(frontLeftMiddlePt, rearLeftMiddlePt ,[1] ,output_parameters = np.array([alpha]))

    geo.assemble()
    geo.evaluate()

    # To line up the z-coordinates with the FrontLeft3 Nacelle
    tempPoint = np.array([ middleLeftIntrp.physical_coordinates[0][0], 0, 2800.0 / mm2ft])
    tempProj =  geo.project_points(tempPoint, projection_direction = down_direction, projection_targets_names=["front_wing"],plot=False)

    geo.assemble()
    geo.evaluate()

    temp  = ((flnacelle3_thrust[0].physical_coordinates[0][2] - middleLeftIntrp.physical_coordinates[0][2]) / (tempProj[0].physical_coordinates[0][2] - middleLeftIntrp.physical_coordinates[0][2]))
    alpha = 1 - temp 
    axisOrigin = geo.perform_linear_interpolation(tempProj[0], middleLeftIntrp ,[1] ,output_parameters = np.array([alpha]))

    geo.assemble()
    geo.evaluate()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #REAR WING POINTS
    leftRearForwardPoint  = np.array([8277.012, 1536.2898, 4006.5]) / mm2ft
    leftRearBackwardPoint = np.array([9123.5639, 1536.2898, 4006.5]) / mm2ft

    leftRearForwardPt, leftRearForwardPtCoord  = geo.project_points(leftRearForwardPoint, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)
    leftRearBackwardPt, leftRearBackwardPtCoord = geo.project_points(leftRearBackwardPoint, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)
    # leftIntrp   = geo.perform_linear_interpolation(leftRearForwardPt,   leftRearBackwardPt, [1], output_parameters = np.array([0.1]))

    # Compute middle point of left side of rear wing 
    midLeftRearWing = geo.perform_linear_interpolation(leftRearForwardPt, leftRearBackwardPt ,[1] ,output_parameters = np.array([0.5]))
    geo.assemble()
    geo.evaluate()

    # Take Coordinates of middle point of left side of rear wing and find points on top and bottom of that x coordinate for left edge of the wing
    temp_leftTopMidPt = midLeftRearWing.physical_coordinates
    temp_leftTopMidPt[0][2] = 13.6    
    rw_leftTopMidPt = geo.project_points(temp_leftTopMidPt, projection_direction = up_direction, projection_targets_names=["rear_wing"],plot=False)

    temp_leftLowerMidPt = midLeftRearWing.physical_coordinates
    temp_leftLowerMidPt[0][2] = 12.9   
    rw_leftLowerMidPt = geo.project_points(temp_leftLowerMidPt, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)

    # Take Coordinates of middle point of left side of rear wing and find points on top and bottom of that x coordinate for center of the wing
    temp_centerTopMidPt = midLeftRearWing.physical_coordinates
    temp_centerTopMidPt[0][1] = 0.0
    temp_centerTopMidPt[0][2] = 13.6
    rw_centerTopMidPt = geo.project_points(temp_centerTopMidPt, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)

    temp_centerLowerMidPt = midLeftRearWing.physical_coordinates
    temp_centerLowerMidPt[0][1] = 0.0
    temp_centerLowerMidPt[0][2] = 12.92
    rw_centerLowerMidPt = geo.project_points(temp_centerLowerMidPt, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)

    geo.assemble()
    geo.evaluate()

    # Interpolate the left side of rear wing and find the point that lines up with the z coordinate in the nacelle
    temp  = ((rlnacelle1_thrust[0].physical_coordinates[0][2] - rw_leftLowerMidPt[0].physical_coordinates[0][2]) / (rw_leftTopMidPt[0].physical_coordinates[0][2] - rw_leftLowerMidPt[0].physical_coordinates[0][2]))
    alpha = 1 - temp 
    rw_leftNacelleAligned = geo.perform_linear_interpolation(rw_leftTopMidPt[0], rw_leftLowerMidPt[0] ,[1] ,output_parameters = np.array([alpha]))

    # Interpolate the left side of center wing and find the point that lines up with the z coordinate in the nacelle
    temp  = ((rlnacelle1_thrust[0].physical_coordinates[0][2] - rw_centerLowerMidPt[0].physical_coordinates[0][2]) / (rw_centerTopMidPt[0].physical_coordinates[0][2] - rw_centerLowerMidPt[0].physical_coordinates[0][2]))
    alpha = 1 - temp 
    rw_centerNacelleAligned = geo.perform_linear_interpolation(rw_centerTopMidPt[0], rw_centerLowerMidPt[0] ,[1] ,output_parameters = np.array([alpha]))

    geo.assemble()
    geo.evaluate()

    # To line up the z-coordinates with the FrontLeft3 Nacelle
    tempPoint = np.array([ middleLeftIntrp.physical_coordinates[0][0], 0, 4010.0 / mm2ft])
    tempProj =  geo.project_points(tempPoint, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)

    geo.assemble()
    geo.evaluate()

    # To line up the Z-coordinates with the RearLeft1 nacelle 
    temp  = ((rlnacelle1_thrust[0].physical_coordinates[0][2] - middleLeftIntrp.physical_coordinates[0][2]) / (tempProj[0].physical_coordinates[0][2] - middleLeftIntrp.physical_coordinates[0][2]))
    alpha = 1 - temp 
    axisOriginRear = geo.perform_linear_interpolation(tempProj[0], middleLeftIntrp ,[1] ,output_parameters = np.array([alpha]))

    geo.assemble()
    geo.evaluate()

    # temp  = ((leftIntrp.physical_coordinates[0][0] -leftRearBackwardPtCoord[0][0]) / (leftRearForwardPtCoord[0][0] - leftRearBackwardPtCoord[0][0]))
    # alpha = 1 - temp 

    # centerRearForwardPoint  = np.array([8065.374522, 0.0, 4006.])/mm2ft
    # centerRearBackwardPoint = np.array([9123.563922, 0.0, 4006.])/mm2ft

    # centerRearForwardPt, centerRearForwardPtCoord = geo.project_points(centerRearForwardPoint, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)
    # centerRearBackwardPt, centerRearBackwardPtCoord = geo.project_points(centerRearBackwardPoint, projection_direction = down_direction, projection_targets_names=["rear_wing"],plot=False)

    # centerRearIntrp = geo.perform_linear_interpolation(centerRearForwardPt, centerRearBackwardPt,[1] ,output_parameters = np.array([alpha]))


    frontwingRotaxis = geo.subtract_pointsets(flnacelle3_thrust[0], axisOrigin)
    rearwingRotaxis  = geo.subtract_pointsets(rw_leftNacelleAligned, rw_centerNacelleAligned)

    geo.assemble()
    geo.evaluate()

    # Reference Point [Meters]
    VSP2NED = np.array([[-1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, -1.0]])
    refPt = np.array([13.35494603, -4.123772014e-16 , 8.118931759])

    results_dict = dict()
    results_dict['front_left_wing_mesh_METERS_VSP']          = fwing_camber_surface_left.physical_coordinates * ft2meters
    results_dict['rear_left_wing_mesh_METERS_VSP']           = rwing_camber_surface_left.physical_coordinates * ft2meters
    results_dict['front_left_wing_mesh_METERS_CG']           = (fwing_camber_surface_left.physical_coordinates - refPt) * ft2meters
    results_dict['rear_left_wing_mesh_METERS_CG']            = (rwing_camber_surface_left.physical_coordinates - refPt) * ft2meters
    results_dict['front_left_wing_mesh_METERS_NED_CG']       = np.matmul(fwing_camber_surface_left.physical_coordinates - refPt, VSP2NED) * ft2meters
    results_dict['rear_left_wing_mesh_METERS_NED_CG']        = np.matmul(rwing_camber_surface_left.physical_coordinates - refPt, VSP2NED) * ft2meters

    results_dict['front_left_nacelle1_origin_METERS_VSP']    = flnacelle1_thrust[0].physical_coordinates * ft2meters
    results_dict['front_left_nacelle1_origin_METERS_CG']     = (flnacelle1_thrust[0].physical_coordinates - refPt) * ft2meters
    results_dict['front_left_nacelle1_origin_METERS_NED_CG'] = (np.matmul(flnacelle1_thrust[0].physical_coordinates - refPt, VSP2NED) * ft2meters)
    results_dict['front_left_nacelle1_vector_VSP']           = flnacelle1_thrust[1].physical_coordinates
    results_dict['front_left_nacelle1_vector_NED']           = np.matmul(flnacelle1_thrust[1].physical_coordinates, VSP2NED)

    results_dict['front_left_nacelle2_origin_METERS_VSP']    = flnacelle2_thrust[0].physical_coordinates * ft2meters
    results_dict['front_left_nacelle2_origin_METERS_CG']     = (flnacelle2_thrust[0].physical_coordinates - refPt) * ft2meters
    results_dict['front_left_nacelle2_origin_METERS_NED_CG'] = (np.matmul(flnacelle2_thrust[0].physical_coordinates - refPt , VSP2NED) * ft2meters)   
    results_dict['front_left_nacelle2_vector_VSP']           = flnacelle2_thrust[1].physical_coordinates
    results_dict['front_left_nacelle2_vector_NED']           = np.matmul(flnacelle2_thrust[1].physical_coordinates, VSP2NED)
    
    results_dict['front_left_nacelle3_origin_METERS_VSP']    = flnacelle3_thrust[0].physical_coordinates * ft2meters
    results_dict['front_left_nacelle3_origin_METERS_CG']     = (flnacelle3_thrust[0].physical_coordinates - refPt) * ft2meters
    results_dict['front_left_nacelle3_origin_METERS_NED_CG'] = (np.matmul(flnacelle3_thrust[0].physical_coordinates - refPt, VSP2NED) * ft2meters)   
    results_dict['front_left_nacelle3_vector_VSP']           = flnacelle3_thrust[1].physical_coordinates
    results_dict['front_left_nacelle3_vector_NED']           = np.matmul(flnacelle3_thrust[1].physical_coordinates, VSP2NED)

    results_dict['rear_left_nacelle1_origin_METERS_VSP']     = rlnacelle1_thrust[0].physical_coordinates * ft2meters
    results_dict['rear_left_nacelle1_origin_METERS_CG']      = (rlnacelle1_thrust[0].physical_coordinates - refPt) * ft2meters
    results_dict['rear_left_nacelle1_origin_METERS_NED_CG']  = (np.matmul(rlnacelle1_thrust[0].physical_coordinates - refPt, VSP2NED) * ft2meters)  
    results_dict['rear_left_nacelle1_vector_VSP']            = rlnacelle1_thrust[1].physical_coordinates
    results_dict['rear_left_nacelle1_vector_NED']            = np.matmul(rlnacelle1_thrust[1].physical_coordinates, VSP2NED)

    results_dict['front_wing_axis_origin_METERS_VSP']        = axisOrigin.physical_coordinates * ft2meters
    results_dict['front_wing_axis_origin_METERS_CG']         = (axisOrigin.physical_coordinates - refPt) * ft2meters
    results_dict['front_wing_axis_origin_METERS_NED_CG']     = np.matmul(axisOrigin.physical_coordinates - refPt, VSP2NED) * ft2meters
    
    results_dict['front_wing_axis_end_METERS_VSP']           = flnacelle3_thrust[0].physical_coordinates * ft2meters
    results_dict['front_wing_axis_end_METERS_CG']            = (flnacelle3_thrust[0].physical_coordinates - refPt) * ft2meters
    results_dict['front_wing_axis_end_METERS_NED_CG']        = np.matmul(flnacelle3_thrust[0].physical_coordinates - refPt, VSP2NED) * ft2meters

    results_dict['rear_wing_axis_origin_METERS_VSP']         = rw_centerNacelleAligned.physical_coordinates * ft2meters
    results_dict['rear_wing_axis_origin_METERS_CG']          = (rw_centerNacelleAligned.physical_coordinates - refPt) * ft2meters
    results_dict['rear_wing_axis_origin_METERS_NED_CG']      = np.matmul(rw_centerNacelleAligned.physical_coordinates - refPt, VSP2NED) * ft2meters
    
    results_dict['rear_wing_axis_end_METERS_VSP']            = rw_leftNacelleAligned.physical_coordinates * ft2meters
    results_dict['rear_wing_axis_end_METERS_CG']             = (rw_leftNacelleAligned.physical_coordinates - refPt) * ft2meters
    results_dict['rear_wing_axis_end_METERS_NED_CG']         = np.matmul(rw_leftNacelleAligned.physical_coordinates - refPt, VSP2NED) * ft2meters
    
    results_dict['front_wing_rotation_axis']                 = frontwingRotaxis.physical_coordinates / np.linalg.norm(frontwingRotaxis.physical_coordinates)
    results_dict['rear_wing_rotation_axis']                  = rearwingRotaxis.physical_coordinates  / np.linalg.norm(rearwingRotaxis.physical_coordinates)


    frontWingDict = dict()
    frontWingDict['front_mesh_shape']  = (9,3)
    frontWingDict['front_wing_mesh']   = results_dict['front_left_wing_mesh_METERS_CG']
    frontWingDict['fl1_thrust_origin'] = results_dict['front_left_nacelle1_origin_METERS_CG']
    frontWingDict['fl2_thrust_origin'] = results_dict['front_left_nacelle2_origin_METERS_CG']
    frontWingDict['fl3_thrust_origin'] = results_dict['front_left_nacelle3_origin_METERS_CG']
    frontWingDict['fl1_thrust_vector'] = results_dict['front_left_nacelle1_vector_VSP']
    frontWingDict['fl2_thrust_vector'] = results_dict['front_left_nacelle2_vector_VSP']
    frontWingDict['fl3_thrust_vector'] = results_dict['front_left_nacelle3_vector_VSP']
    frontWingDict['rot_axis_origin'] = results_dict['front_wing_axis_origin_METERS_CG']
    frontWingDict['rot_axis_vector'] = results_dict['front_wing_rotation_axis']    
    
    rearWingDict = dict()
    rearWingDict['rear_mesh_shape']  = (9,3)
    rearWingDict['rear_wing_mesh']   = results_dict['rear_left_wing_mesh_METERS_CG']
    rearWingDict['rl1_thrust_origin'] = results_dict['rear_left_nacelle1_origin_METERS_CG']
    rearWingDict['rl1_thrust_vector'] = results_dict['rear_left_nacelle1_vector_VSP']
    rearWingDict['rot_axis_origin'] = results_dict['rear_wing_axis_origin_METERS_CG']
    rearWingDict['rot_axis_vector'] = results_dict['rear_wing_rotation_axis']    

    plotMeshesAndVectors(frontWingDict, rearWingDict, 1)

    # vp = Plotter()
    # vps_list = []
    # points = fwing_camber_surface_left.physical_coordinates * ft2meters
    # vps = Points(points, c='red', r=4)
    # vps_list.append(vps)
    # points = (fwing_camber_surface_left.physical_coordinates - refPt) * ft2meters
    # vps = Points(points, c='blue', r=4)
    # vps_list.append(vps)
    # vp.show(vps_list, 'Projection Results', axes=1, viewup="z", interactive=True)

    return results_dict