import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vedo import Points, Plotter, LegendBox

def plot_3d_array(inArr):
    t_dim, numPts, coords = inArr.shape
    vp = Plotter()
    vps_list = []

    for i in np.arange(t_dim):
        points = inArr[i,:,:]
        vps = Points(points, c='red', r=4)
        vps_list.append(vps)

    vp.show(vps_list, 'Projection Results', axes=1, viewup="z", interactive=True)

