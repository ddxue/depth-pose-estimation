import numpy as np
import os

root_dir = '/mnt0/emma/shotton/shotton_people/'
data_dir = [root_dir + d + '/' for d in os.listdir(root_dir) if d.find('person_') != -1]
SIZE = 500

def parse_data(view):
  for i in range(len(data_dir)):
    X = np.load(data_dir[i]+view+'_X.npy')
    labels = np.load(data_dir[i]+view+'_labels.npy')
    num_images = int(X[-1][3]+1)
    print 'Num images:', num_images
    num_split = num_images / SIZE + 1
    print 'Num splits:', num_split
    for j in range(num_split):
      dir_path = data_dir[i] + '0' + str(j) + '/'
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)
      if j != num_split - 1:
          start = np.where(X[:,3] == j * SIZE)[0][0]
          end = np.where(X[:,3] == (j+1) * SIZE)[0][-1]
          X_split = X[start:end,:]
          label_split = labels[start:end]
      else:
          start = np.where(X[:,3] == j * SIZE)[0][0]
          X_split = X[start:,:]
          label_split = labels[start:]
      print X_split[0][3], X_split[-1][3]
      np.save(dir_path+view+'_X.npy', X_split)
      np.save(dir_path+view+'_labels.npy', label_split)


def main_0():
  views = ['side', 'top']
  for view in views:
    parse_data(view)

def main():
  data_dir = '/mnt0/data/ITOP/out/'
  out_root = '/mnt0/emma/shotton/shotton_people/'
  num_people = 12

  for i in range(num_people):
    prefix = str(i).zfill(2)
    joints_top_path = [data_dir + d for d in os.listdir(data_dir) if d.find(prefix+'_joints_top') != -1]
    joints_side_path = [data_dir + d for d in os.listdir(data_dir) if d.find(prefix+'_joints_side') != -1]
    print 'Saving', joints_top_path[0], 'to', out_root+'person_'+prefix
    joints_top = np.load(joints_top_path[0])
    print 'Saving', joints_side_path[0], 'to', out_root+'person_'+prefix
    joints_side = np.load(joints_side_path[0])
    np.save(out_root+'person_'+prefix+'/side_joints.npy', joints_side)
    np.save(out_root+'person_'+prefix+'/top_joints.npy', joints_top)

if __name__ == "__main__":
  main()
