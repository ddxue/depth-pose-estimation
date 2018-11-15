import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.externals import joblib
from multiprocessing import Process as worker
from get_acc_joints import *
import util
import os
import glob

# parameters
num_joints = 15
root_dir = '/mnt0/emma/shotton/shotton_people/'
BATCH_SIZE = 500
train_batch = 3
test_batch = 3
ensemble = 8

def get_data_ensemble(data_dir, train_batch):
  data = {}
  dirs = [data_dir+d+'/' for d in os.listdir(data_dir) if d.find('0') != -1]
  dirs.sort()
  print dirs

  num_batch = min(train_batch, len(dirs))
  features = []
  labels = []
  for i in range(num_batch):
    print 'Load data from', dirs[i]
    features.append(np.load(dirs[i] + 'f16.npy'))
    labels.append(np.load(dirs[i] + 'labels.npy').reshape(features[-1].shape[0], 1))
  features = np.vstack(features)
  labels = np.vstack(labels)
  data['features'] = features
  data['labels'] = labels.reshape(labels.shape[0],)
  return data

def train_rf(i, root_dir):
  data_dir = root_dir + 'person_' + str(i).zfill(2) + '/'
  train_data = get_data_ensemble(data_dir, train_batch)

  # train a random forest classifier
  print 'Feature shape:', train_data['features'].shape
  print 'Label shape:', train_data['labels'].shape
  rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
  rf.fit(train_data['features'], train_data['labels'])

  joblib.dump(rf, root_dir + 'models/rf_' + str(i).zfill(2) + '.pkl')
  print root_dir + 'models/rf_' + str(i).zfill(2) + '.pkl saved'

def test_rf(root_dir, test_range, train_range):
  # processes = []
  for i in test_range:
      test_rf_batch(i, train_range, root_dir)
  #   processes.append(
  #     worker(
  #       target=test_rf_batch,
  #       name="Thread #%d" % i,
  #       args=(i, train_range, root_dir)
  #     )
  #   )
  # [t.start() for t in processes]
  # [t.join() for t in processes]

def test_rf_batch(i, train_range, root_dir):
  data_dir = root_dir + 'person_' + str(i).zfill(2) + '/'
  test_data = get_data_ensemble(data_dir, test_batch)
  print 'Loading test batch from', data_dir

  pred_prob_ensemble = np.zeros((len(test_data['labels']), num_joints))
  for j in train_range:
    rf = joblib.load(root_dir + 'models/rf_' + str(j).zfill(2) + '.pkl')
    print 'Model loaded for', root_dir + 'models/rf_' + str(j).zfill(2) + '.pkl'
    pred_prob_ensemble += rf.predict_proba(test_data['features'])

  pred_label = np.argmax(pred_prob_ensemble, axis=1)
  np.save(data_dir+'pred_prob.npy', pred_prob_ensemble)
  np.save(data_dir+'pred_label.npy', pred_label)

  avg_accuracy = util.get_cm_acc(test_data['labels'].astype(int), pred_label)
  print 'MAP for person', i, ':', avg_accuracy

  accuracy = 0.0
  accClass = np.zeros((num_joints,2))
  for j in range(pred_label.shape[0]):
    accClass[int(test_data['labels'][j])][1] += 1
    if pred_label[j] == test_data['labels'][j]:
      accClass[pred_label[j]][0] += 1
      accuracy += 1
  accuracy /= float(pred_label.shape[0])

  for j in range(num_joints):
    print 'Person', i, 'Class', j, ':', accClass[j][0], '/', accClass[j][1], accClass[j][0]/accClass[j][1]
  print 'Person', i, 'Total accuracy:', accuracy


def main():
  view = 'top'
  root_dir = '/mnt0/emma/shotton/shotton_people_' + view + '/'
  train_range = range(5, 12)
  test_range = range(4)
  mode = 'train'

  if mode == 'train':
    # processes = []
    for i in train_range:
        train_rf(i, root_dir)
    #   processes.append(
    #     worker(
    #       target=train_rf,
    #       name="Thread #%d" % i,
    #       args=(i, root_dir)
    #     )
    #   )
    # [t.start() for t in processes]
    # [t.join() for t in processes]
  elif mode == 'test':
    test_rf(root_dir, test_range, train_range)

if __name__ == "__main__":
  main()
