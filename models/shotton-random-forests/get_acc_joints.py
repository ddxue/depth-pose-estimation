import numpy as np
from util import *

# true_joints: 500 x 14 x 3
# output: 14 x 1 (accuracy for each joint)
def get_joint_acc(true_joints, pred_joints, true_joint_count, person_dir, pixel_thred=100, offset=10):
  # if a joint has less than 100 pixels, count as occluded
  mask = np.array(true_joint_count > pixel_thred, dtype=int) # (300, 8)
  true_num_image = np.sum(mask, axis=0) # (8,)
  dist = np.sqrt(np.sum(pow((true_joints - pred_joints)/10, 2), axis=2)) # convert mm to cm
  result = np.array(dist < offset, dtype=int) * mask
  result = np.sum(result, axis=0)
  accuracy = result.astype(float) / true_num_image
  # np.save(person_dir + 'local_error.npy', dist)
  print true_num_image
  return accuracy

def pixel2world2(coord, W=320, H=240, C=3.506667e-3):
  coordW = np.zeros(coord.shape)
  coordW[:,0] = (coord[:,0] - 0.5 * W) * coord[:,2] * C
  coordW[:,1] = -(coord[:,1] - 0.5 * H) * coord[:,2] * C
  coordW[:,2] = coord[:,2]
  return coordW

def get_joint_count(X, labels, num_data, offset, num_joints):
  joint_count = np.zeros((num_data, num_joints))
  for i in range(num_data):
    start = np.where(X[:,3] == i + offset)[0][0]
    end = np.where(X[:,3] == i + offset)[0][-1]
    for j in range(num_joints):
      joint_count[i, j] = len(np.where(labels[start:end] == j)[0])
  return joint_count

def gaussian_density(X, pred_prob, num_joints, b=0.065, push_back=0.039):
  new_prob = np.zeros(pred_prob.shape)
  depth = X[:,2].reshape(len(X),1) / 1000 + push_back # mm -> m
  new_prob = pred_prob * depth**2
  coord_world = pixel2world2(X[:,:3])
  density = np.zeros(pred_prob.shape)
  for i in range(len(coord_world)):
    density[i] = np.sum(new_prob * np.exp(-np.sum(((coord_world - coord_world[i])/1000/b)**2, axis=1)).reshape(len(new_prob),1), axis=0)
    density[i] /= np.sum(density[i])
  return density

def main_0():
  root_dir = '/mnt0/emma/shotton/'
  test_data = root_dir + 'shotton_people/person_08/00/'
  out_prob = root_dir + 'itop_out/ensemble_prob_1_500.npy'
  out_label = root_dir + 'itop_out/ensemble_label_1_500.npy'
  density_path = test_data + 'density_500.npy'
  num_images = 500
  offset = 0
  num_joints = 15
  view = 'side'

  X = np.load(test_data + view + '_X.npy')
  true_label = np.load(test_data + view + '_labels.npy')
  # true_joints: x_pixel, y_pixel, z, x_world, y_world
  true_joints = np.load(test_data + view + '_joints.npy')[:,:num_joints,2:]
  col = [1, 2, 0]
  true_joints = true_joints[:,:,col]
  pred_prob, pred_label = np.load(out_prob), np.load(out_label)
  pred_joints = part_to_joint(X, pred_label, pred_prob, num_images, offset, num_joints, density_path)
  np.save(test_data + view + '_pred_joints.npy', pred_joints)

  true_joint_count = get_joint_count(X, true_label, num_images, offset, num_joints)
  total_acc = get_joint_acc(true_joints, pred_joints, true_joint_count)
  avg_acc = np.mean(total_acc)
  print 'Estimate from', density_path
  print 'Precision per joint:', total_acc
  print 'Mean average accuracy:', avg_acc
  print '###########################################'

def main():
    view = 'side'
    num_joints = 15
    root_dir = '/mnt0/emma/shotton/shotton_people_' + view + '/'
    test_range = range(8,9)

    for i in test_range:
        prefix = str(i).zfill(2)
        person_dir = root_dir + 'person_' + prefix + '/'
        density_path = person_dir + 'density.npy'
        X = np.load(person_dir + 'X.npy')
        true_label = np.load(person_dir + 'labels.npy')
        num_images = int(X[-1][3]+1)
        offset = 0
        true_joints = np.load(person_dir + 'joints.npy')[:,:num_joints,2:]
        col = [1, 2, 0]
        true_joints = true_joints[:,:,col]
        pred_prob, pred_label = np.load(person_dir+'pred_prob.npy'), np.load(person_dir+'pred_label.npy')
        pred_joints = part_to_joint(X, pred_label, pred_prob, num_images, offset, num_joints, density_path)
        np.save(person_dir+'pred_joints.npy', pred_joints)
        true_joint_count = get_joint_count(X, true_label, num_images, offset, num_joints)
        total_acc = get_joint_acc(true_joints, pred_joints, true_joint_count, person_dir)
        avg_acc = np.mean(total_acc)
        print 'Estimate from', density_path
        print 'Precision per joint:', total_acc
        print 'Mean average accuracy:', avg_acc
        print '###########################################'


if __name__ == "__main__":
  main()
