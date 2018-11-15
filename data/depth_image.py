import sys, os
from copy import deepcopy
import math

import numpy as np
import imageio
from scipy import misc
from scipy.optimize import least_squares

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d import Axes3D

rotation_matrix = np.array([[ 0.97477055,  0.14537687, -0.16937515],
 [ 0.14537687,  0.16231096,  0.97597168],
 [ 0.16937515, -0.97597168,  0.13708151]])

kResX = 320
kResY = 240

torso_idx = [0, 1, 2, 3]
left_arm_idx = [2, 4, 5, 6, 7]
right_arm_idx = [2, 8, 9, 10, 11]
left_leg_idx = [0, 12, 13, 14, 15]
right_leg_idx = [0, 16, 17, 18, 19]
body_group_idx = [left_leg_idx, right_leg_idx, torso_idx, left_arm_idx, right_arm_idx]

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print((int(event.xdata), int(event.ydata)), ",")
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

class DepthImage(object):
    '''
    Class for dealing with depth images and their associated
    3D point clouds.
    '''

    def __init__(self, im_path):
        self.load_kinect_defaults()
        self.depth_im = self.load_depth_from_img(im_path)
        self.skeleton_pixels = None
        self.skeleton_coords = None

    '''
    LOAD_DEPTH_FROM_IMG load depth image given image path.
    '''
    def load_depth_from_img(self, depth_path):
        depth_im = imageio.imread(depth_path) # im is a numpy array
        return depth_im

    def display_depth_img(self, overlay_skeleton=False):
        plt.imshow(self.depth_im, cmap='gray')
        plt.show()

    def load_skeleton_coords(self, coord_list):
        self.skeleton_coords = coord_list

    def load_skeleton_pixels(self, pixel_list):
        self.skeleton_pixels = pixel_list

    def display_skeleton_img(self):
        if not self.skeleton_pixels:
            return

        # Plot joint points
        x_s, y_s = zip(*self.skeleton_pixels)
        plt.scatter(x_s, y_s)

        # Plot skeleton segments
        for body_idx in body_group_idx:
            x, y = zip(*[self.skeleton_pixels[joint_idx] for joint_idx in body_idx])
            plt.plot(x, y)

        plt.imshow(self.depth_im, cmap='gray')
        # plt.show()

    def draw_skeleton_img(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        line, = ax.plot([0], [0])  # empty line
        linebuilder = LineBuilder(line)

        ax.imshow(self.depth_im, cmap='gray')
        plt.show()

    def display_skeleton_point_cloud(self):
        skeleton_coords = []
        for joint_idx, (x_s, y_s) in enumerate(self.skeleton_pixels):
            if self.depth_im[y_s, x_s] == 0:
                print("Failure at JointID: ", joint_idx)
            z = self.depth_im[y_s, x_s] / 256.0 # note im is indexed by [row, col] == [y, x]
            x = z * (x_s - self.cx) / self.fx
            y = z * (y_s - self.cy) / self.fy
            skeleton_coords.append((x, y, z))
        x, y, z = zip(*skeleton_coords)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z)

        for body_idx in body_group_idx:
            x, y, z = zip(*[skeleton_coords[joint_idx] for joint_idx in body_idx])
            plt.plot(x, y, z)

        plt.axis('scaled')
        plt.axis('off')
        plt.show()

    '''
    LOAD_KINECT_DEFAULTS set the camera intrinsics to some Kinect default.
    '''
    def load_kinect_defaults(self):
        self.fx, self.fy = 570.0, 570.0
        self.cx, self.cy = 320.0, 240.0

        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        self.inv_K = np.linalg.inv(self.K)

    '''
    POINT_CLOUD transforms a depth image into a point cloud  with
    one point for each pixel in the image, using the camera transform
    for a camera centred at (cx, cy) with field of view (fx, fy).

    Arguments:
        depth - depth is a 2-D ndarray with shape (rows, cols) containing
        depths from 1 to 254 inclusive.
    Returns:
        point_3d - 3-D array with shape (rows, cols, 3). Pixels with invalid
        depth in the input have NaN for the z-coordinate in the result
    '''
    def generate_point_cloud(self):
        # Generate row and column coordinates for each pixel
        rows, cols = self.depth_im.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

        valid = (self.depth_im > 0) & (self.depth_im < 255) # mask array of valid pixels

        z = np.where(valid, self.depth_im / 256.0, np.nan)
        x = np.where(valid, z * (c - self.cx) / self.fx, 0)
        y = np.where(valid, z * (r - self.cy) / self.fy, 0)

        point_cloud = np.dstack((x, y, z))
        return point_cloud

    def display_point_cloud(self, subsample = 3000):
        point_cloud = self.generate_point_cloud()
        h, w, xwy = point_cloud.shape
        h_idx = np.random.choice(h, subsample)
        w_idx = np.random.choice(w, subsample)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(point_cloud[h_idx,w_idx,0], point_cloud[h_idx,w_idx,1], point_cloud[h_idx,w_idx,2], c='red')

        skeleton_coords = []
        for joint_idx, (x_s, y_s) in enumerate(self.skeleton_pixels):
            if self.depth_im[y_s, x_s] == 0:
                print("Failure at JointID: ", joint_idx)
            z = self.depth_im[y_s, x_s] / 256.0 # note im is indexed by [row, col] == [y, x]
            x = z * (x_s - self.cx) / self.fx
            y = z * (y_s - self.cy) / self.fy
            skeleton_coords.append((x, y, z))
        x, y, z = zip(*skeleton_coords)
        ax.scatter(x, y, z)


        for body_group in [torso, left_arm, right_arm, left_leg, right_leg]:
            x, y, z = zip(*body_group)
            ax.plot(x, y, z)

        plt.axis('scaled')
        plt.axis('off')
        plt.show()

    """
    Projection formulas as described in
    https://groups.google.com/forum/#!msg/unitykinect/1ZFCHO9PpjA/1KdxUTdq90gJ
    Given (x,y,z) coordinates, converts that point into its (x, y) pixel
    coordinate in the 2D image.
    """
    def pixel_from_coords(self, x, y, z):
        kRealWorldXtoZ = 1.122133
        kRealWorldYtoZ = 0.84176

        fCoeffX = kResX / kRealWorldXtoZ
        fCoeffY = kResY / kRealWorldYtoZ

        xPixel = (fCoeffX * float(x) / float(z)) + (kResX / 2)
        yPixel = (kResY / 2) - (fCoeffY * float(y) / float(z))

        return int(xPixel), int(yPixel)

    def rotate_point(self, x, y, z):
        point = np.array([x, y, z])
        new_point = rotation_matrix.dot(point)
        return new_point[0], new_point[1], new_point[2]

    def coords_from_pixel(self, xPixel, yPixel):
        z = self.depth_im[yPixel, xPixel] / 256.0
        x = z * (xPixel - self.cx) / self.fx
        y = z * (yPixel - self.cy) / self.fy
        return x, y, z
