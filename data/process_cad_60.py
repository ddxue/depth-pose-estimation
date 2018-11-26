'''
Process Cornell Human Activies Dataset (CAD-60)

http://pr.cs.cornell.edu/humanactivities/data.php
'''

import numpy as np
import os

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


import imageio
import matplotlib.pyplot as plt

# Depth image dimension
kResX = 320
kResY = 240

# List of joint names
JOINT_NAMES = ['HEAD', 'NECK', 'TORSO', \
            'LEFT_SHOULDER', 'LEFT_ELBOW', \
            'RIGHT_SHOULDER', 'RIGHT_ELBOW', \
             'LEFT_HIP', 'LEFT_KNEE', \
             'RIGHT_HIP', 'RIGHT_KNEE', \
             'LEFT_HAND', 'RIGHT_HAND', \
             'LEFT_FOOT', 'RIGHT_FOOT', \
            ]

"""
Projection formulas as described in
https://groups.google.com/forum/#!msg/unitykinect/1ZFCHO9PpjA/1KdxUTdq90gJ.
Given (x,y,z) coordinates, converts that point into its x pixel number in the 2D image.
"""
def pixel_from_coords(self, x, y, z):
    kRealWorldXtoZ = 1.122133
    kRealWorldYtoZ = 0.84176

    fCoeffX = kResX / kRealWorldXtoZ
    fCoeffY = kResY / kRealWorldYtoZ

    xPixel = (fCoeffX * float(x) / float(z)) + (kResX / 2)
    yPixel = (kResY / 2) - (fCoeffY * float(y) / float(z))

    return int(xPixel), int(yPixel)

C = 3.8605e-3 #NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

def pixel2world(pixel, C):
    world = np.empty(pixel.shape)
    world[:, 0] = (pixel[:, 0] - kResX / 2.0) * C * pixel[:, 2]
    world[:, 1] = -(pixel[:, 1] - kResY /2.0) * C * pixel[:, 2]
    world[:, 2] = pixel[:, 2]
    return world

def world2pixel(world, C):
    pixel = np.empty(world.shape)
    pixel[:, 0] = np.rint(world[:, 0] / world[:, 2] / C + kResX / 2.0)
    pixel[:, 1] = np.rint(-world[:, 1] / world[:, 2] / C + kResY / 2.0)
    pixel[:, 2] = world[:, 2]
    return pixel

def read_cad60_skels(folder_name='.'):
    result = {}
    for dirName, subdirList, fileList in os.walk(folder_name, topdown=False):
        # print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname.endswith('.txt') and fname[0] != 'a':
                with open(os.path.join(dirName, fname)) as f:
                    content = [line.rstrip() for line in f.readlines()] # strip whitespace
                    content = content[:-1] # ignore END line
                    all_coords = []
                    for line in content:
                        frame_num, skeleton_coords = parse_skeleton_text(line)
                        all_coords.append(skeleton_coords)
                    video_id = os.path.splitext(fname)[0]
                    result[video_id] = all_coords
    return result

def parse_skeleton_text(line):
    # Parse line by comma
    fields = line.split(',')
    assert len(fields) == 172, 'Actual length is: ' + str(len(fields))

    frame_num = fields[0]
    skeleton_coords = []

    offset = 1
    for joint_id in range(1, 16): # 1, 2,...,11
        if joint_id <= 11: # 1, 2,...,11
            offset += 10  # skip orientation and conf

        x = float(fields[offset])
        offset += 1
        y = float(fields[offset])
        offset += 1
        z = float(float(fields[offset]))
        offset += 1
        conf = float(fields[offset])
        offset += 1

        # pixel_x, pixel_y = pixel_from_coords(x, y, z)
        skeleton_coords.append([x, y, z])

    assert len(skeleton_coords) == 15, 'Actual length is: ' + str(len(output))
    skeleton_coords = [
        skeleton_coords[1],
        skeleton_coords[0],
        skeleton_coords[3],
        skeleton_coords[4],
        skeleton_coords[11],
        skeleton_coords[5],
        skeleton_coords[6],
        skeleton_coords[12],
        skeleton_coords[8],
        skeleton_coords[13],
        skeleton_coords[10],
        skeleton_coords[14],
        skeleton_coords[7],
        skeleton_coords[9],
        skeleton_coords[2],
    ]

    skeleton_coords = np.array(skeleton_coords)
    return frame_num, skeleton_coords # ",".join((str(v) for v in output))

def process_cad60_imgs(folder_name, skeletons):
    depth_images = []
    joints = []

    for dirName, subdirList, fileList in os.walk(folder_name, topdown=False):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname.startswith('Depth_'):
                video_id = os.path.basename(os.path.normpath(dirName))
                frame_num = int(os.path.splitext(fname)[0].split('_')[1]) - 1

                # print("Video ID: ", video_id)
                # print("Frame #: ", frame_num)

                depth_path = os.path.join(dirName, fname)
                rgb_path = os.path.join(dirName, fname)

                depth_im = imageio.imread(depth_path) # H x W = depth_z

                skeleton_frames = skeletons[video_id]
                joint_coords = skeleton_frames[frame_num] # = world_x, world_y, depth_z

                # Check non-zero depth values
                if np.count_nonzero(joint_coords[:,2] == 0) > 0:
                    print(joint_coords[:,2])
                    continue

                # Convert to pixel (im_y, im_x, depth_z) coordinates
                joint_coords = world2pixel(joint_coords, C) # im_y, im_x, depth_z

                # Add to array
                depth_images.append(depth_im) # H x W = depth_z
                joints.append(joint_coords)


            # [n x height (240) width (320)]
            # [n x joints (15) x 3]

    depth_images, joints = np.array(depth_images), np.array(joints)
    return depth_images, joints

dir_ = 'datasets/CAD-60/'

skeletons = read_cad60_skels(dir_)
depth_images, joints = process_cad60_imgs(dir_, skeletons)
print(depth_images.shape)
print(joints.shape)

np.save('processed/CAD-60/depth_images.npy', depth_images)
np.save('processed/CAD-60/joints.npy', joints)
