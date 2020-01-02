"""
Class for plotting a quadrotor

Author: Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
from PIL import Image

from matplotlib import pyplot as pltq
pltq.rcParams["animation.html"] = "jshtml"

class Quadrotor():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.51, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.show_animation = show_animation

        if self.show_animation:
            self.fg = pltq.figure()
            self.ax = self.fg.add_subplot(111, projection='3d')

        self.update_pose(x, y, z, roll, pitch, yaw)
        
    def clear_data(self):
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
    
        if self.show_animation:
            self.plot()
            
        # draw the renderer
        self.fg.canvas.draw()

        image_from_plot = np.frombuffer(self.fg.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(self.fg.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])


    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        #plt.cla()
        self.ax.clear()
        x = [p1_t[0], p2_t[0], p3_t[0], p4_t[0]]
        y = [p1_t[1], p2_t[1], p3_t[1], p4_t[1]]
        z = [p1_t[2], p2_t[2], p3_t[2], p4_t[2]]
        
        self.ax.plot3D(x, y, z,'k.')
        
        
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')
        

        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:')

        lim = 5
        self.ax.axis(xmin=-lim,xmax=lim,ymin=-lim,ymax=lim)

        self.ax.set_zlim(0, 10)
        
        #plt.pause(0.001)      
        