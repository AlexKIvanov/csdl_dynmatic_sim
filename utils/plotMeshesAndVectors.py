import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D

def plotMeshesAndVectors(frontWingDict, rearWingDict, nt):

    frontMeshShape      = frontWingDict['front_mesh_shape']
    frontWingMesh       = frontWingDict['front_wing_mesh']
    frontThrustOrigin1  = frontWingDict['fl1_thrust_origin']
    frontThrustOrigin2  = frontWingDict['fl2_thrust_origin']
    frontThrustOrigin3  = frontWingDict['fl3_thrust_origin']                                    
    frontThrustVector1  = frontWingDict['fl1_thrust_vector']
    frontThrustVector2  = frontWingDict['fl2_thrust_vector']
    frontThrustVector3  = frontWingDict['fl3_thrust_vector']
    frontWingAxisRotOrigin = frontWingDict['rot_axis_origin']
    frontWingAxisRotVector = frontWingDict['rot_axis_vector']

    rearMeshShape      = rearWingDict['rear_mesh_shape']
    rearWingMesh          = rearWingDict['rear_wing_mesh']
    rearThrustOrigin1     = rearWingDict['rl1_thrust_origin']
    rearThrustVector1     = rearWingDict['rl1_thrust_vector']
    rearWingAxisRotOrigin = rearWingDict['rot_axis_origin']
    rearWingAxisRotVector = rearWingDict['rot_axis_vector']
    
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    if nt == 1:
        if frontMeshShape == (1,3,3,3):
            frontWingMesh = np.reshape(frontWingMesh, (3,3,3))
            rearWingMesh  = np.reshape(rearWingMesh, (3,3,3))

        ax.scatter(frontWingMesh[:,0], frontWingMesh[:,1], frontWingMesh[:,2], c='r')
        ax.scatter(frontThrustOrigin1[0,0], frontThrustOrigin1[0,1], frontThrustOrigin1[0,2], c='b')
        ax.scatter(frontThrustOrigin2[0,0], frontThrustOrigin2[0,1], frontThrustOrigin2[0,2], c='b')
        ax.scatter(frontThrustOrigin3[0,0], frontThrustOrigin3[0,1], frontThrustOrigin3[0,2], c='b')
        ax.quiver(frontThrustOrigin1[0,0], frontThrustOrigin1[0,1], frontThrustOrigin1[0,2], frontThrustVector1[0,0], frontThrustVector1[0,1], frontThrustVector1[0,2], color='g')
        ax.quiver(frontThrustOrigin2[0,0], frontThrustOrigin2[0,1], frontThrustOrigin2[0,2], frontThrustVector2[0,0], frontThrustVector2[0,1], frontThrustVector2[0,2], color='g')
        ax.quiver(frontThrustOrigin3[0,0], frontThrustOrigin3[0,1], frontThrustOrigin3[0,2], frontThrustVector3[0,0], frontThrustVector3[0,1], frontThrustVector3[0,2], color='g')
        ax.quiver(frontWingAxisRotOrigin[0,0], frontWingAxisRotOrigin[0,1], frontWingAxisRotOrigin[0,2], frontWingAxisRotVector[0,0], frontWingAxisRotVector[0,1], frontWingAxisRotVector[0,2], color='k')
        
        ax.scatter(rearWingMesh[:,0], rearWingMesh[:,1], rearWingMesh[:,2], c='r')
        ax.scatter(rearThrustOrigin1[0,0], rearThrustOrigin1[0,1], rearThrustOrigin1[0,2], c='b')
        ax.quiver(rearThrustOrigin1[0,0], rearThrustOrigin1[0,1], rearThrustOrigin1[0,2], rearThrustVector1[0,0], rearThrustVector1[0,1], rearThrustVector1[0,2], color='g')
        ax.quiver(rearWingAxisRotOrigin[0,0], rearWingAxisRotOrigin[0,1], rearWingAxisRotOrigin[0,2], rearWingAxisRotVector[0,0], rearWingAxisRotVector[0,1], rearWingAxisRotVector[0,2], color='k')
        plt.show()

    else:
        for i in np.arange(nt):
            if frontMeshShape == (nt,3,3,3):
                frontWingMeshTemp = np.reshape(frontWingMesh[i,:,:,:], (3,3,3))
                rearWingMeshTemp  = np.reshape(rearWingMesh[i,:,:,:], (3,3,3))
                frontWingMeshMod  = np.reshape(frontWingMeshTemp, (9,3))
                rearWingMeshMod  = np.reshape(rearWingMeshTemp, (9,3))
                
            else:
                frontWingMeshMod = frontWingMesh
                rearWingMeshMod  = rearWingMesh

            ax.scatter(frontWingMeshMod[:,0], frontWingMeshMod[:,1], frontWingMeshMod[:,2], c='r')
            ax.scatter(frontThrustOrigin1[i,0], frontThrustOrigin1[i,1], frontThrustOrigin1[i,2], c='b')
            ax.scatter(frontThrustOrigin2[i,0], frontThrustOrigin2[i,1], frontThrustOrigin2[i,2], c='b')
            ax.scatter(frontThrustOrigin3[i,0], frontThrustOrigin3[i,1], frontThrustOrigin3[i,2], c='b')
            ax.quiver(frontThrustOrigin1[i,0], frontThrustOrigin1[i,1], frontThrustOrigin1[i,2], frontThrustVector1[i,0], frontThrustVector1[i,1], frontThrustVector1[i,2], color='g')
            ax.quiver(frontThrustOrigin2[i,0], frontThrustOrigin2[i,1], frontThrustOrigin2[i,2], frontThrustVector2[i,0], frontThrustVector2[i,1], frontThrustVector2[i,2], color='g')
            ax.quiver(frontThrustOrigin3[i,0], frontThrustOrigin3[i,1], frontThrustOrigin3[i,2], frontThrustVector3[i,0], frontThrustVector3[i,1], frontThrustVector3[i,2], color='g')
            ax.quiver(frontWingAxisRotOrigin[i,0], frontWingAxisRotOrigin[i,1], frontWingAxisRotOrigin[i,2], frontWingAxisRotVector[i,0], frontWingAxisRotVector[i,1], frontWingAxisRotVector[i,2], color='k')
            
            ax.scatter(rearWingMeshMod[:,0], rearWingMeshMod[:,1], rearWingMeshMod[:,2], c='r')
            ax.scatter(rearThrustOrigin1[i,0], rearThrustOrigin1[i,1], rearThrustOrigin1[i,2], c='b')
            ax.quiver(rearThrustOrigin1[i,0], rearThrustOrigin1[i,1], rearThrustOrigin1[i,2], rearThrustVector1[i,0], rearThrustVector1[i,1], rearThrustVector1[i,2], color='g')
            ax.quiver(rearWingAxisRotOrigin[i,0], rearWingAxisRotOrigin[i,1], rearWingAxisRotOrigin[i,2], rearWingAxisRotVector[i,0], rearWingAxisRotVector[i,1], rearWingAxisRotVector[i,2], color='k')
        plt.show()
