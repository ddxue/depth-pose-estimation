 import numpy as np
from multiprocessing import Process as worker
import os
import glob
import argparse

def map_features(X, theta_u, theta_v, images, thread_index, feature_dir):
    normalize = 100
    eps = 1e-7
    m = X.shape[0]
    num_features = theta_u.shape[0]
    features = np.zeros((m,num_features))
    for i in range(0,m):
        index = X[i][3]
        width = images[index].shape[0]
        height = images[index].shape[1]
        if (i % 100 == 0):
          print 'Processed:', i, '/', m
        for j in range(0,num_features):
            left = X[i][:2] + theta_u[j] / (X[i][2]/normalize+eps)
            right =  X[i][:2] + theta_v[j] / (X[i][2]/normalize+eps)

            left_new = np.maximum(0, np.minimum(left, [width-1, height-1]))
            right_new = np.maximum(0, np.minimum(right, [width-1, height-1]))

            features[i][j] = float(images[index][left_new[0], left_new[1]]) - float(images[index][right_new[0], right_new[1]])
            # if (i % 100 == 0):
            #  print(' left: ' + str(left_new) + ' right: ' + str(right_new))
            #  print('image[i][left]: ' + str(images[index][left_new[0], left_new[1]]) + ' image[i][right]: ' + str(images[index][right_new[0], right_new[1]]))
    features = features.astype(np.float16)
    np.save(feature_dir + str(thread_index) + '.npy', features)
    return features

def map_features_thread(X, theta_u, theta_v, images, data_dir, view, num_threads):
  X_split = np.array_split(X, num_threads)
  processes = []

  feature_dir = data_dir + 'features/'
  print feature_dir
  if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)
    print feature_dir
  files = glob.glob(feature_dir+'*')
  for f in files:
    os.remove(f)

  for i in range(num_threads):
    processes.append(
      worker(
        target = map_features,
        name="Thread #%d" % i,
        args=(X_split[i], theta_u, theta_v, images, i, feature_dir)
      )
    )
  [t.start() for t in processes]
  [t.join() for t in processes]

  feature_all = []
  for i in range(num_threads):
    feature = np.load(feature_dir + str(i) + '.npy')
    feature_all.append(feature)
  feature_all = np.vstack(feature_all).astype(np.float16)
  np.save(data_dir + 'f16_' + view + '.npy', feature_all)

def main_0():
  root_dir = '/mnt0/emma/shotton/data_ext/'
  data_dir = ['data2/']
  data_dir = [root_dir + d for d in data_dir]
  max_offset = 100
  num_features = 2000

  # phi = (theta, tau)
  theta_u = np.random.randint(-max_offset, max_offset, size=(num_features,2)) # each row is (u1,u2)
  theta_v = np.random.randint(-max_offset, max_offset, size=(num_features,2)) # each row is (v1,v2)

  for directory in data_dir:
    images = np.load(directory+'images.npy')
    dirs = os.listdir(directory)
    # dirs = [directory+d+'/' for d in dirs if d.find('0') != -1]
    dirs = [directory+d+'/' for d in ['01', '05', '07']]
    for d in dirs:
      X  = np.load(d + 'X.npy')
      print "Loading X from", d
      map_features_thread(X, theta_u, theta_v, images, d, num_threads=35)

def main(**kwargs):
  root_dir = '/scail/scratch/group/vision/bypeng/healthcare/shotton/shotton_people/'
  person_id = kwargs.get('person_id', 0)
  person_id = str(person_id).zfill(2)
  data_dir = root_dir + 'person_' + person_id + '/'
  max_offset = 100
  num_features = 2000
  views = ['top']

  # phi = (theta, tau)
  # theta_u = np.random.randint(-max_offset, max_offset, size=(num_features,2))
  # theta_v = np.random.randint(-max_offset, max_offset, size=(num_features,2))
  theta_u = np.load('theta_u.npy')
  theta_v = np.load('theta_v.npy')

  for view in views:
    images = np.load(data_dir + view + '_images.npy')
    dirs = [data_dir+d+'/' for d in os.listdir(data_dir) if d.find('0') != -1]
    dirs.sort()
    for d in dirs:
      print 'Loading from', d, 'View', view
      X = np.load(d + view + '_X.npy')
      map_features_thread(X, theta_u, theta_v, images, d, view, num_threads=8)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--person_id', type=int, required=True)
  args = parser.parse_args()
  main(**vars(args))
