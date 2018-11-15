import numpy as np
import re
from os import listdir, path
import os
from sklearn.metrics import confusion_matrix
from multiprocessing import Process as worker
from get_acc_joints import *

X_path = 'X.npy'
images_path = 'images.npy'
labels_path = 'labels.npy'
width = 320
height = 240

def load_data_new(file_list, person_id, npy_root):
  depth_top, depth_side = [], []
  # label_top, label_side = [], []
  # joint_top, joint_side = [], []

  depth_top_files = [f for f in file_list if f.find('depth_top') != -1]
  depth_side_files = [f for f in file_list if f.find('depth_side') != -1]
  # label_top_files = [f for f in file_list if f.find('label_top') != -1]
  # label_side_files = [f for f in file_list if f.find('label_side') != -1]
  # joints_top_files = [f for f in file_list if f.find('joints_top') != -1]
  # joints_side_files = [f for f in file_list if f.find('joints_side') != -1]

  num_data = len(depth_top_files)

  for i in range(num_data):
    if i % 100 == 0:
      print 'Thread', person_id, 'Processed', i, '/', num_data
    depth_top.append(np.loadtxt(depth_top_files[i], delimiter='\n').reshape(height, width))
    depth_side.append(np.loadtxt(depth_side_files[i], delimiter='\n').reshape(height, width))
    # label_top.append(np.loadtxt(label_top_files[i], delimiter='\n').reshape(height, width))
    # label_side.append(np.loadtxt(label_side_files[i], delimiter='\n').reshape(height, width))
    # joint_top.append(np.loadtxt(joints_top_files[i], delimiter=','))
    # joint_side.append(np.loadtxt(joints_side_files[i], delimiter=','))

  root = npy_root + person_id + '_'
  print 'Saving to', root

  depth_top = np.array(depth_top)
  np.save(root+'depth_top.npy', depth_top)
  del depth_top
  depth_side = np.array(depth_side)
  np.save(root+'depth_side.npy', depth_side)
  del depth_side
  # label_top = np.array(label_top)
  # np.save(root+'label_top.npy', label_top)
  # del label_top
  # label_side = np.array(label_side)
  # np.save(root+'label_side.npy', label_side)
  # del label_side
  # joint_top = np.array(joint_top)
  # np.save(root+'joint_top.npy', joint_top)
  # del joint_top
  # joint_side = np.array(joint_side)
  # np.save(root+'joint_side.npy', joint_side)
  # del joint_side

def load_data(data_dir, out_file, num_images):
  depth = []
  label = []

  for i in range(num_images):
    depth.append(np.loadtxt(data_dir +'depth'+str(i)+'.dat', delimiter='\n').reshape(height, width))
    label.append(np.loadtxt(data_dir +'label'+str(i)+'.dat', delimiter='\n').reshape(height, width))
    if i % 100 == 0:
      print 'Processed', i, 'data'

  depth = np.array(depth)
  label = np.array(label)
  data = np.empty((depth.shape[0], height, width, 2))
  data[:, :, :, 0] = depth
  data[:, :, :, 1] = label

  np.save(out_file, data)
  print('Data saved!')
  return data
  # return (depth, label)

def load_joints(data_dir, out_file, num_images, num_joints):
  if path.isfile(out_file):
    joints = np.load(out_file)
    print 'Joint exists!'
    return joints

  joints = []
  for i in range(num_images):
    joints.append(np.loadtxt(data_dir +'joints-top'+str(i)+'.dat', delimiter=','))
    if i % 100 == 0:
      print 'Processed', i, 'data'

  joints = np.array(joints)[:,:num_joints,:]
  np.save(out_file, joints)
  return joints

def processData(image, labelAll, numJoint, pixelPerJoint, out_path, view):
  out_X = out_path + view + '_' + X_path
  out_images = out_path + view + '_' + images_path
  out_labels = out_path + view + '_' + labels_path

  numImage = len(image)

  X = []
  label = []
  depth = []
  index_image = []

  for i in range(numImage):
    # each joint should have roughly the same number of pixels
    jointNumPixel = np.zeros((numJoint))
    for j in range(numJoint):
      jointNumPixel[j] = np.nonzero(labelAll[i] == j)[0].shape[0] # joint label: 0,...,13

    jointNumPixel = np.minimum(pixelPerJoint, jointNumPixel).astype(int)
    print 'Num pixel per joint', jointNumPixel
    print 'Total pixel', np.sum(jointNumPixel)
    for j in range(numJoint):
      pair = np.nonzero(labelAll[i] == j)
      indices = np.column_stack((pair[0], pair[1]))
      depth_val = image[i][indices[:,0], indices[:,1]]

      if len(indices) > 0:
        random_sample = np.random.choice(len(indices), jointNumPixel[j], replace=False)
        random_sample.sort()
      else:
        random_sample = np.array([]).astype(int)

      X += indices[random_sample, :].tolist()
      depth += depth_val[random_sample].tolist()
      index_image += (i * np.ones((jointNumPixel[j]))).tolist()
      label += np.extract(labelAll[i] == j, labelAll[i])[random_sample].tolist()
      # print index_image[-1]

  X = np.array(X)
  depth = np.array(depth)
  depth = depth.reshape(depth.shape[0], 1)
  index_image = np.array(index_image)
  index_image = index_image.reshape(index_image.shape[0], 1)
  np.save(out_path+'index.npy', index_image)

  print 'X', X.shape
  print 'depth', depth.shape
  print 'index', index_image.shape

  # each row of X: (x,y,depth,#image)
  X = np.append(X, depth, axis=1)
  X = np.append(X, index_image, axis=1)
  label = np.array(label).astype(int)
  print 'label', label.shape
  np.save(out_X, X)
  np.save(out_images, image)
  np.save(out_labels, label)
  print 'Saving', out_X, out_images, out_labels
  return (image, X, label)

def part_to_joint(X, label, prob, num_data, offset, num_joints, density_path, lam=0.14, push_back=39):
  joints = np.zeros((num_data, num_joints, 3))
  if path.isfile(density_path):
    density = np.load(density_path)
  else:
    density = np.zeros(prob.shape)

  col = [1,0,2]
  coord_world = pixel2world2(X[:,:3][:,col])
  coord_world[:,2] += push_back
  for i in range(num_data):
    if i % 10 == 0:
      print 'Processed:', i
    start = np.where(X[:,3] == i + offset)[0][0]
    end = np.where(X[:,3] == i + offset)[0][-1]
    if path.isfile(density_path):
      new_prob = density[start:end]
    else:
      new_prob = gaussian_density(X[start:end], prob[start:end], num_joints)
      density[start:end] = new_prob
    for j in range(num_joints):
      prob_thred = new_prob[:,j] * np.array(new_prob[:,j] > lam, dtype=int)
      if np.sum(prob_thred) == 0:
        joints[i, j] = [0, 0, 1000]
      else:
        joints[i, j] = np.sum(coord_world[start:end] * prob_thred.reshape(len(prob_thred), 1), axis=0) / (np.sum(prob_thred))
  np.save(density_path, density)
  return joints

def partition_data(data, pixel_per_joint, num_joints, train_ratio):
  (images, X, labels) = processData(data, num_joints, pixel_per_joint)
  num_pixels = X.shape[0]
  num_pixel_train = round(num_pixels * train_ratio)

  labels_train = labels[:num_pixel_train]
  labels_test = labels[num_pixel_train:]

  X_train = X[:num_pixel_train]
  X_test = X[num_pixel_train:]

  return (X_train, X_test, labels_train, labels_test)

def get_cm_acc(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1)
  num_classes = cm_normalized.shape[0]
  avg_accuracy = np.trace(cm_normalized) / num_classes
  return avg_accuracy

def get_joints(data, pred_prob, num_joints, pixel_per_joint, train_ratio):
  (X_train, X_test, labels_train, labels_test) = partition_data(data, pixel_per_joint, num_joints, train_ratio)
  num_data = data.shape[0]
  joints = part_to_joint(X_test, labels_test, pred_prob, num_data, num_joints)
  return joints

def main_0():
  data_ext = '/mnt0/emma/shotton/data_ext/'
  data_roots = ['data2/']
  data_roots = [data_ext + d for d in data_roots]
  data_dirs = ['data2/data_2/']
  data_dirs = [data_ext + d for d in data_dirs]
  out_files = ['data2/data2.npy']
  out_files = [data_ext + f for f in out_files]
  out_joints = ['data2/joints.npy']
  out_joints = [data_ext + f for f in out_joints]
  for i in range(len(data_dirs)):
    if not path.isfile(out_files[i]):
      load_data(data_dir=data_dirs[i], out_file=out_files[i], num_images=4800)
      # load_joints(data_dir=data_dirs[i], out_file=out_joints[i], num_images=4800, num_joints=8)
    else:
      data = np.load(out_files[i])
      processData(data, numJoint=8, pixelPerJoint=500, out_path=data_roots[i])

def main():
  data_root = '/scail/scratch/group/vision/bypeng/healthcare/shotton/shotton_people/'
  npy_root = '/scail/scratch/group/vision/hospital/ITOP/'
  views = ['side', 'top']
  num_joints = {'side':15, 'top':15}
  num_pixel_per_joint = {'side':200, 'top':300}
  num_people = 12
  range1 = range(8, 12) 

  for view in views:
    processes = []
    # for i in range(num_people):
    for i in range1:
      index = str(i).zfill(2)
      image = np.load(npy_root+index+'_depth_'+view+'.npy')
      label = np.load(npy_root+index+'_predicts_'+view+'.npy')
      dir_path = data_root + 'person_' + index + '/'
      print 'Load', npy_root+index+'_depth_'+view+'.npy', 'Save to', dir_path
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)
      processes.append(
        worker(
          target = processData,
          name="Thread #%d" % i,
          args=(image, label, num_joints[view], num_pixel_per_joint[view], dir_path, view)
        )
      )
    [t.start() for t in processes]
    [t.join() for t in processes]

def main_1():
  data_root = '/mnt0/data/ITOP/all/'
  npy_root = '/mnt0/data/ITOP/out/'
  data_files = listdir(data_root)
  num_people = 12

  processes = []
  for i in range(num_people):
    index = str(i).zfill(2)
    person_i = [data_root+f for f in data_files if \
                len(re.findall(r"\b"+index+"_", f)) != 0]
    person_i.sort()
    print 'Processing person', i, 'Num frames', len(person_i)
    processes.append(
      worker(
        target=load_data_new,
        name="Thread #%d" % i,
        args=(person_i, index, npy_root)
      )
    )
  [t.start() for t in processes]
  [t.join() for t in processes]

def save_random():
  num_features = 2000
  max_offset = 100
  
  # phi = (theta, tau)
  theta_u = np.random.randint(-max_offset, max_offset, size=(num_features,2))
  theta_v = np.random.randint(-max_offset, max_offset, size=(num_features,2))

  np.save('theta_u.npy', theta_u)
  np.save('theta_v.npy', theta_v)


if __name__ == "__main__":
  save_random()
