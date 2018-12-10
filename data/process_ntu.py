import numpy as np
import imageio
import os

# Depth image dimension
kResX = 424
kResY = 512

C = 3.8605e-3 # is this still the same in this dataset?

def world2pixel(world, C):
	pixel = np.empty(world.shape)
	pixel[:, 0] = np.rint(world[:, 0] / world[:, 2] / C + kResX / 2.0)
	pixel[:, 1] = np.rint(-world[:, 1] / world[:, 2] / C + kResY / 2.0)
	pixel[:, 2] = world[:, 2]
	return pixel

# Map from joint names to index
JOINT_IDX = {
    'NECK': 0,
    'HEAD': 1,
    'LEFT SHOULDER': 2,
    'LEFT ELBOW': 3,
    'LEFT HAND': 4,
    'RIGHT SHOULDER': 5,
    'RIGHT ELBOW': 6,
    'RIGHT HAND': 7,
    'LEFT KNEE': 8,
    'LEFT FOOT': 9,
    'RIGHT KNEE': 10,
    'RIGHT FOOT': 11,
    'LEFT HIP': 12,
    'RIGHT HIP': 13,
    'TORSO': 14,
}

def rearrange_joints(joints):
	"""Takes in 25 xyz joints and returns 15 xyz joints"""
	new_joints = []

	for i, joint in enumerate(joints):
		new_joints = [
			joints[20], # neck
			joints[3], # head
			joints[4], # left shoulder
			joints[5], # left elbow
			joints[7], # left hand
			joints[8], # right shoulder
			joints[9], # right elbow
			joints[11], # right hand
			joints[13], # left knee
			joints[14], # left foot
			joints[17], # right knee
			joints[18], # right foot
			joints[12], # left hip
			joints[16], # right hip
			joints[0], # torso
		]

	return new_joints

# aggregate files to go through
names = []
for filename in os.listdir('datasets/NTU-RGBD/nturgb+d_depth_masked/'):
	if filename.startswith("S00"):
		names.append(filename)

joints = []
depth_images = []

count = 100
for name in sorted(names): # example name: 'S001C001P001R001A001'
	if count == 0:
		break

	# aggrgating joints
	f = open('datasets/NTU-RGBD/nturgb+d_skeletons/' + name + '.skeleton')
	for i, line in enumerate(f):
		if i == 0:
			framecount = int(line.strip())
			print('num frames: ', framecount)
			curr_joints = []
			prev = 1
			denom = 1
		elif i == prev:
			num_skeleton = int(line.strip())
			denom = num_skeleton * 27 + 1 # 1 row for num_skeleton, (1 + 1 + 25) for (1 metadata + 1 num_joints + 25 joints per skeleton)
		elif i - 1 == prev: # skipping row if metadata
			continue
		elif i - 2 == prev: # num joints, should always be 25 -- this doesn't check on additional skeletons
			num_joints = int(line.strip())
			if num_joints != 25:
				assert False, 'num_joints = {}'.format(num_joints)
		else: # the 25 joints
			if (i - 28 - prev) < 0: # only aggregate joints on first skeleton
				joint_info = line.strip().split(" ") # should always be 12 long
				if len(joint_info) != 12:
					assert False, 'len joint_info = {}'.format(len(joint_info))
				xyz = [float(x) for x in joint_info[:3]]
				curr_joints.append(xyz)
			if i == denom + prev - 1: # end of frame
				curr_joints = rearrange_joints(curr_joints)

				curr_joints_transformed = world2pixel(np.array(curr_joints), C)
				curr_joints_transformed[:,2] = curr_joints_transformed[:,2] / 1000.0 # convert from mm to meters -- do we still need to do this?
				joints.append(curr_joints_transformed)
				curr_joints = []
				prev+=denom

	# aggregating corresponding depth images - does this include mul skeleton?
	for i in range(1, framecount + 1):
		num_zeros = 8 - len(str(i))
		suffix = '0'*num_zeros + str(i)
		depth_im = imageio.imread('datasets/NTU-RGBD/nturgb+d_depth_masked/' + name + '/MDepth-' + suffix + '.png')
		if depth_im.shape != (424, 512): # H x W should be 424 x 512
			assert False, 'depth image shape {}'.format(depth_im.shape)
		depth_im = depth_im / 1000.0 # convert from mm to meters -- do we still need this?
		depth_images.append(depth_im)

	count -= 1

joints = np.array(joints)
depth_images = np.array(depth_images)

print('joints: ', joints.shape)
print('depth images: ', depth_images.shape)

np.save('processed/NTU-RGBD/depth_images.npy', depth_images)
np.save('processed/NTU-RGBD/joints.npy', joints)
