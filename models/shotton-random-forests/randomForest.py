import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from get_acc_joints import *
import util
import os

# parameters
numJoints = 8
root_dir = '/mnt0/emma/shotton/data_ext/'
data_dirs = ['data1/']
data_dirs = [root_dir + d for d in data_dirs]
BATCH_SIZE = 500
train_batch = 1
test_batch = 1
ensemble1 = 3
ensemble2 = 0

def get_data_ensemble(data_dir, num_ensemble):
  data = {}
  dirs = os.listdir(data_dir)
  dirs = [data_dir+d+'/' for d in dirs if d.find('0') != -1]
  dirs.sort()
  features = []
  labels = []
  for j in range(train_batch):
    index = train_batch * num_ensemble + j
    print 'Load training data from', dirs[index]
    features.append(np.load(dirs[index] + 'f16.npy'))
    labels.append(np.load(dirs[index] + 'labels.npy').reshape(features[-1].shape[0], 1))
  features = np.vstack(features)
  labels = np.vstack(labels)
  data['features'] = features
  data['labels'] = labels.reshape(labels.shape[0],)
  return data 

def get_data(batch, train_flag, root_dir=root_dir, data_dir=data_dirs):
  data = {}
  features = []
  labels = []
  for directory in data_dir:
    dirs = os.listdir(directory)
    dirs = [directory+d+'/' for d in dirs if d.find('0') != -1]
    dirs.sort()
    if train_flag:
      del dirs[-1] # reserve the last one for testing
      if batch > len(dirs):
        batch -= len(dirs)
        for d in dirs:
          labels.append(np.load(d + 'labels.npy'))
	  features.append(np.load(d + 'f16.npy'))
          print d, 'train loaded!'
      else:
        for i in range(batch):
	  labels.append(np.load(dirs[i] + 'labels.npy'))
	  features.append(np.load(dirs[i] + 'f16.npy'))
	  print i, 'train loaded!'
        break
    else:
      if batch > 0:
        labels.append(np.load(dirs[-1] + 'labels.npy'))
        features.append(np.load(dirs[-1] + 'f16.npy'))
        print 'Load test data from', dirs[-1]
        batch -= 1
      else:
        break
  # features = np.vstack(features)
  length = [len(f) for f in features]
  feature_stack = np.zeros((sum(length), 2000))
  label_stack = np.zeros((sum(length),))
  start = 0
  for i in range(len(length)):
    feature_stack[start:start+length[i]:] = features[i]
    label_stack[start:start+length[i]:] = labels[i]
    start += length[i]
  data['features'], data['labels'] = feature_stack, label_stack
  return data

test_data = get_data(batch=test_batch, train_flag=False)    
ensemble_prob = np.zeros((len(test_data['labels']), numJoints))

ensemble = ensemble1 + ensemble2
num_train_images = train_batch * (ensemble1+ensemble2) * BATCH_SIZE
print 'Num ensembles:', ensemble1, ensemble2, 'Num images:', num_train_images

for i in range(ensemble1): 
  train_data = {}
  train_data = get_data_ensemble(data_dirs[0], i)

  # for j in range(numJoints):
   # num = np.nonzero(train_data['labels'] == j)[0].shape[0]
   # print('Joint ' + str(j) + ': ' + str(num))

  # train a random forest classifier
  print 'Feature shape:', train_data['features'].shape
  print 'Label shape:', train_data['labels'].shape
  rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
  rf.fit(train_data['features'], train_data['labels'])

  tree_estimator = rf.estimators_
  for j in range(len(tree_estimator)):
    t = tree_estimator[j].tree_
    tree.export_graphviz(t, out_file=data_dirs[0]+'trees/tree'+str(j)+'.dot')

  del train_data

  # prediction
  pred_label = rf.predict(test_data['features'])
  pred_prob = rf.predict_proba(test_data['features'])
  ensemble_prob += pred_prob

  # accuracy
  score = rf.score(test_data['features'], test_data['labels'])
  print('Ensemble ' + str(i) +' Score: ' + str(score))

for i in range(ensemble2): 
  train_data = {}
  train_data = get_data_ensemble(data_dirs[-1], i)

  print 'Feature shape:', train_data['features'].shape
  print 'Label shape:', train_data['labels'].shape
  rf = RandomForestClassifier(n_estimators=3, criterion='entropy', max_depth=20)
  rf.fit(train_data['features'], train_data['labels'])

  del train_data

  # prediction
  pred_label = rf.predict(test_data['features'])
  pred_prob = rf.predict_proba(test_data['features'])
  ensemble_prob += pred_prob

  # accuracy
  score = rf.score(test_data['features'], test_data['labels'])
  print('Ensemble ' + str(i) +' Score: ' + str(score))

ensemble_label = np.argmax(ensemble_prob, axis=1)
avg_accuracy = util.get_cm_acc(test_data['labels'].astype(int), ensemble_label)
print('Avg accuracy: ' + str(avg_accuracy))

pred_label_path = root_dir + 'out/ensemble2_label_'+str(ensemble)+'_'+str(num_train_images)+'.npy'
pred_prob_path = root_dir + 'out/ensemble2_prob_'+str(ensemble)+'_'+str(num_train_images)+'.npy'
np.save(pred_label_path, ensemble_label)
np.save(pred_prob_path, ensemble_prob)

accuracy = 0.0
accClass = np.zeros((numJoints,2))
for i in range(0,ensemble_label.shape[0]):
  accClass[int(test_data['labels'][i])][1] += 1
  if ensemble_label[i] == test_data['labels'][i]:
    accClass[ensemble_label[i]][0] += 1
    accuracy += 1
accuracy /= float(ensemble_label.shape[0])

for i in range(0,numJoints):
  print 'Class', i, ':', accClass[i][0], '/', accClass[i][1], accClass[i][0]/accClass[i][1] 
print('Total accuracy: ' + str(accuracy))  
print '##################################'
