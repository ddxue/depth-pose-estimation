import sys
import argparse

import numpy as np
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Process, Queue

from helper import *

"""
Directory Structure:

depth-pose-estimation/
    data/
        process_depth_images.py
        ...
        datasets/
            ...
        processed/
            CAD-60/
                depth_images.npy
                joints.npy
                theta.npy
                body_centers.npy
            NTU-RGBD/
                depth_images.npy
                joints.npy
                ...
            ...
    models/
        random-tree-walks/
            rtw.py
            helper.py
        ...
    ...
"""

###############################################################################
# Parser arguments
###############################################################################

parser = argparse.ArgumentParser(description='Random Tree Walks algorithm.')

# Loading options for the model and data
parser.add_argument('--load-params', action='store_true',
                    help='Load the parameters')
parser.add_argument('--load-model', action='store_true',
                    help='Load a pretrained model')
parser.add_argument('--load-test', action='store_true',
                    help='Run trained model on test set')

# Location of data directories
parser.add_argument('--input-dir', type=str, default='../../data/processed',
                    help='Directory of the processed input')
parser.add_argument('--dataset', type=str, default='CAD-60',
                    help='Name of the dataset to load')

# Location of saved data directories
parser.add_argument('--model-dir', type=str, default='../../output/random-tree-walks/models',
                    help='Directory of the saved model')
parser.add_argument('--png-dir', type=str, default='../../output/random-tree-walks/png',
                    help='Directory to save prediction images')
parser.add_argument('--preds-dir', type=str, default='../../output/random-tree-walks/preds',
                    help='Directory to save predictions')

# Training options
parser.add_argument('--num-steps', type=int, default=300,
                    help='Random seed')
parser.add_argument('--seed', type=int, default=1111,
                    help='Random seed')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Shuffle the data')
parser.add_argument('--multithread', action='store_true',
                    help='Train each joint on a separate threads')

# Output options
parser.add_argument('--make-png', action='store_true',
                    help='Draw predictions on top of inputs')

args = parser.parse_args()

###############################################################################
# Constants
###############################################################################

# Number of joints in a skeleton
NUM_JOINTS = 15

# List of joint names
JOINT_NAMES = ['NECK', 'HEAD', \
                'LEFT SHOULDER', 'LEFT ELBOW', 'LEFT HAND', \
                'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT HAND', \
                'LEFT KNEE', 'LEFT FOOT', \
                'RIGHT KNEE', 'RIGHT FOOT', \
                'LEFT HIP', \
                'RIGHT HIP', \
                'TORSO']

H = 240
W = 320

C = 3.8605e-3 # NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

kinem_order  = [0, 1, 2, 5, 3, 6, 4, 7, 8, 10, 9, 11]
kinem_parent = [-1, 0, 0, 0, 2, 5, 3, 6, -1, -1, 8, 10]

###############################################################################
# Training hyperparameters
###############################################################################

###############################################################################
# Load dataset splits
###############################################################################

def load_dataset(processed_dir):
    """
    Loads the processed dataset.
    """
    logger.debug('Loading data from directory %s', processed_dir)

    # Load input and labels from numpy files
    depth_images = np.load(os.path.join(processed_dir, 'depth_images.npy')) # N x H x W depth images
    joints = np.load(os.path.join(processed_dir, 'joints.npy')) # N x NUM_JOINTS x 3 joint locations

    # Load and apply mask to the depth images
    # depth_mask = np.load(os.path.join(processed_dir, '/depth_mask.npy')) # N x H x W depth mask
    # depth_images = depth_images * depth_mask

    # Load parameters from disk
    # theta = np.load(os.path.join(processed_dir, '/theta.npy'))
    # body_centers = np.load(os.path.join(processed_dir, '/body_centers.npy'))

    return depth_images, joints #, theta, body_centers

def compute_params(joints, num_feats=500, maxOffFeat=150):
    """
    @params:
        maxOffFeat : the maximum offset for features (before divided by d)
        num_feats : the number of features of each offset point
    """
    logger.debug('Computing parameters...')

    # Calculate the body centers
    left_hip = (joints[:,2] + 2 * joints[:,8]) / 3.0
    right_hip = (joints[:,5] + 2 * joints[:,10]) / 3.0
    body_centers = (joints[:,2] + left_hip + joints[:,5] + right_hip) / 4.0

    # Compute the theta = (-maxOffFeat, maxOffFeat)
    theta = np.random.randint(-maxOffFeat, maxOffFeat + 1, (4, num_feats)) # (4, num_feats)

    # # Save parameters to disk
    # np.save(processed_dir + '/body_centers.npy', body_centers)
    # np.save(processed_dir + '/theta.npy', theta)
    return body_centers, theta

def split_dataset(X, y, body_centers, train_ratio=0.20):
    """
    Splits the dataset according to the train-test ratio.

    @ params:
        X : depth images (N x H x W)
        y : joint positions (N x NUM_JOINTS x 3)
        train_ratio : ratio of training to test
    """
    test_ratio = 1.0 - train_ratio
    num_test = int(X.shape[0] * test_ratio)

    X_train, y_train = X[num_test:], y[num_test:]
    X_test, y_test = X[:num_test], y[:num_test]
    body_centers_train, body_centers_test = body_centers[num_test:], body_centers[:num_test]

    logger.debug('Data loaded: # training data: %d, # test data: %d', X_train.shape[0], X_test.shape[0])
    return X_train, y_train, X_test, y_test, body_centers_train, body_centers_test

processed_dir = os.path.join(args.input_dir, args.dataset) # directory of saved numpy files
depth_images, joints = load_dataset(processed_dir)
body_centers, theta = compute_params(joints)
X_train, y_train, X_test, y_test, body_centers_train, body_centers_test = split_dataset(depth_images, joints, body_centers)

num_train = X_train.shape[0]
num_test = X_test.shape[0]

###############################################################################
# Train model
###############################################################################

def get_features(img, theta, q, z, large_num=100):
    img[img == 0] = large_num

    coor = q[:2][::-1] # coor: y, x
    coor[0] = np.clip(coor[0], 0, H-1) # y
    coor[1] = np.clip(coor[1], 0, W-1) # x
    coor = np.rint(coor).astype(int)
    dq = z if img[tuple(coor)] == large_num else img[tuple(coor)]

    x1 = np.clip(coor[1] + theta[0] / dq, 0, W-1).astype(int)
    x2 = np.clip(coor[1] + theta[2] / dq, 0, W-1).astype(int)

    y1 = np.clip(coor[0] + theta[1] / dq, 0, H-1).astype(int)
    y2 = np.clip(coor[0] + theta[3] / dq, 0, H-1).astype(int)

    feature = img[y1, x1] - img[y2, x2]
    return feature

def get_training_samples(joint_id, X, y, body_centers, theta, num_feats=500, num_samples=500, max_offset_xy=60, max_offset_z=2):
    '''Creates the training samples for each joint.

    Each sample is (i, q, u, f) where:
         i is the index of the depth image,
         q is the random offset point,
         u is the unit direction vector toward the joint location,
         f is the feature array

    @params
        num_samples : number of samples of each joint
        max_offset_xy : maximum offset for samples in (x, y) axes
        max_offset_z : maximum offset for samples in z axis
    '''
    num_train, _, _ = X.shape

    S_u = np.zeros((num_train, num_samples, 3), dtype=np.float64)
    S_f = np.zeros((num_train, num_samples, num_feats), dtype=np.float64)

    for train_idx in range(num_train):
        if train_idx % 100 == 0:
            logger.debug('Joint %s: Processing image %d / %d', JOINT_NAMES[joint_id], train_idx, num_train)

        for sample_idx in range(num_samples):
            offset_xy = np.random.randint(-max_offset_xy, max_offset_xy + 1, 2)
            offset_z = np.random.uniform(-max_offset_z, max_offset_z, 1)
            offset = np.concatenate((offset_xy, offset_z))

            S_u[train_idx, sample_idx] = 0 if np.linalg.norm(offset) == 0 else (-offset / np.linalg.norm(offset))
            S_f[train_idx, sample_idx] = get_features(X[train_idx], theta, y[train_idx] + offset, body_centers[train_idx][2])

    return S_u, S_f

def stochastic(regressor, features, unit_directions, K=20):
    L = {}

    indices = regressor.apply(features) # leaf id of each sample
    leaf_ids = np.unique(indices) # array of unique leaf ids

    logger.debug('Running stochastic (minibatch) K-Means...')
    for leaf_id in leaf_ids:
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000)
        labels = kmeans.fit_predict(unit_directions[indices == leaf_id])
        weights = np.bincount(labels).astype(float) / labels.shape[0]

        # Normalize the centers
        centers = kmeans.cluster_centers_
        centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
        # checkUnitVectors(centers)

        L[leaf_id] = (weights, centers)
    return L

def train(joint_id, X, y, model_dir, min_samples_leaf=400):
    regressor_path = os.path.join(model_dir, 'regressor' + str(joint_id) + '.pkl')
    L_path = os.path.join(model_dir, 'L' + str(joint_id) + '.pkl')

    # if loadModels and os.path.isfile(regressor_path) and os.path.isfile(L_path):
    #     logger.debug('Loading model %s from files...', JOINT_NAMES[jointID])
    #     regressor = pickle.load(open(regressor_path, 'rb'))
    #     L = pickle.load(open(L_path, 'rb'))
    # else:
    logger.debug('Start training %s model...', JOINT_NAMES[joint_id])

    regressor = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)

    X_reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y_reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

    rows = np.logical_not(np.all(X_reshape == 0, axis=1))
    regressor.fit(X_reshape[rows], y_reshape[rows])

    logger.debug('Model %s - valid samples: %d / %d', JOINT_NAMES[joint_id], X_reshape[rows].shape[0], X_reshape.shape[0])

    leaf_ids = regressor.apply(X_reshape)
    bin = np.bincount(leaf_ids)
    unique_ids = np.unique(leaf_ids)
    biggest = np.argmax(bin)
    smallest = np.argmin(bin[bin != 0])

    logger.debug('Model %s - # Leaves: %d', JOINT_NAMES[joint_id], unique_ids.shape[0])
    logger.debug('Model %s - Biggest leaf id: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], biggest, bin[biggest], np.sum(bin))
    logger.debug('Model %s - Smallest leaf id: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], smallest, bin[bin != 0][smallest], np.sum(bin))
    logger.debug('Model %s - Average leaf size: %d', JOINT_NAMES[joint_id], np.sum(bin) / unique_ids.shape[0])

    L = stochastic(regressor, X_reshape, y_reshape)

    # Save models to disk
    pickle.dump(regressor, open(regressor_path, 'wb'))
    pickle.dump(L, open(L_path, 'wb'))

    return regressor, L

def train_parallel(joint_id, X, y, body_centers, theta, model_dir, regressor_queue, L_queue):
    S_u, S_f = get_training_samples(joint_id, X, y, body_centers, theta)
    regressor, L = train(joint_id, S_f, S_u, model_dir)
    regressor_queue.put({joint_id: regressor})
    L_queue.put({joint_id: L})

def train_series(joint_id, X, y, body_centers, theta, model_dir):
    S_u, S_f = get_training_samples(joint_id, X, y, body_centers, theta)
    regressor, L = train(joint_id, S_f, S_u, model_dir)
    return regressor, L

logger.debug('\n------- Training models -------')

regressors, Ls = {}, {}
if not args.multithread:
    for joint_id in range(NUM_JOINTS):
        regressors[joint_id], Ls[joint_id] = train_series(joint_id, X_train, y_train[:, joint_id], body_centers_train, theta, args.model_dir)
else:
    processes = []
    regressor_queue, L_queue = Queue(), Queue()

    for joint_id in range(NUM_JOINTS):
        p = Process(target=train_parallel, name='Thread #%d' % joint_id, args= \
                    (joint_id, X_train, y_train[:, joint_id], body_centers_train, theta, args.model_dir, regressor_queue, L_queue))
        processes.append(p)
        p.start()

    regressors_tmp = [regressor_queue.get() for p in processes]
    Ls_tmp = [L_queue.get() for p in processes]

    regressors = dict(i.items()[0] for i in regressors_tmp)
    Ls = dict(i.items()[0] for i in Ls_tmp)

    [p.join() for p in processes]

###############################################################################
# Evaluate model
###############################################################################

def test_model(regressor, L, theta, qm0, img, body_center, num_steps=args.num_steps, step_size=3):
    qm = np.zeros((num_steps+1, 3))
    qm[0] = qm0
    joint_pred = np.zeros(3)

    for i in range(num_steps):
        f = get_features(img, theta, qm[i], body_center[2]).reshape(1, -1)
        leaf_id = regressor.apply(f)[0]

        # L[leaf_id][0]: weights, L[leaf_id][1]: centers
        idx = np.random.choice(K, p=L[leaf_id][0])
        u = L[leaf_id][1][idx]

        qm[i+1] = qm[i] + u * step_size
        qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1)
        qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1)
        qm[i+1][2] = img[int(qm[i+1][1]), int(qm[i+1][0])]
        joint_pred += qm[i+1]

    joint_pred = joint_pred / num_steps
    return qm, joint_pred

logger.debug('\n------- Testing models -------')

qms = np.zeros((num_test, NUM_JOINTS, args.num_steps+1, 3))
y_pred = np.zeros((num_test, NUM_JOINTS, 3))
local_error = np.zeros((num_test, args.num_steps+1, NUM_JOINTS, 3))

# if loadTest:
#     qms = np.load(outDir+modelsDir+'/qms.npy')
#     y_pred = np.load(outDir+modelsDir+'/y_pred.npy')
#     localErr = np.load(outDir+modelsDir+'/local_err.npy')
# else:
for kinem_idx, joint_id in enumerate(kinem_order):
    logger.debug('Testing %s model', JOINT_NAMES[joint_id])
    for test_idx in range(num_test):
        qm0 = body_centers_test[test_idx] if kinem_parent[kinem_idx] == -1 else y_pred[test_idx][kinem_parent[kinem_idx]]
        qms[test_idx][joint_id], y_pred[test_idx][joint_id] = test_model(regressors[joint_id], Ls[joint_id], theta, qm0, I_test[test_idx], body_centers_test[test_idx])
        local_error[test_idx, :, joint_id, :] = y_test[test_idx, joint_id] - qms[test_idx][joint_id]

y_pred[:, :, 2] = y_test[:, :, 2]

# np.save(modelsDir + 'qms.npy', qms)
# np.save(modelsDir + 'y_pred.npy', y_pred)
# np.save(modelsDir + 'local_error.npy', local_error)
#
# mkdir(outDir + modelsDir + '/pred/')
# for jointID in range(NUM_JOINTS):
#     # print(y_test[:, jointID].shape)
#     np.savetxt(outDir+modelsDir+'/pred/'+JOINT_NAMES[jointID]+'_test.txt', y_test[:, jointID], fmt='%.3f')
#     # print(y_pred[:, jointID].shape)
#     np.savetxt(outDir+modelsDir+'/pred/'+JOINT_NAMES[jointID]+'_pred.txt', y_pred[:, jointID], fmt='%.3f ')

###############################################################################
# Run evaluation metrics
###############################################################################

def get_dists(y_test, y_pred):
    assert y_test.shape == y_pred.shape
    dists = np.zeros((y_test.shape[:2]))

    for i in range(y_test.shape[0]):
        p1 = pixel2world(y_test[i], C)
        p2 = pixel2world(y_pred[i], C)
        dists[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))
    return dists

dists = get_dists(y_test, y_pred) * 100.0

# np.savetxt(outDir+modelsDir+'/pred/dists.txt', dists, fmt='%.3f')

#dists_pixel = np.zeros((y_test.shape[:2]))
#for i in range(y_test.shape[0]):
#    p1 = y_test[i]
#    p2 = y_pred[i]
#    dists_pixel[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))

mAP = 0
for i in range(NUM_JOINTS):
    logger.debug('\nJoint %s:', JOINT_NAMES[i])
    logger.debug('Average distance: %f cm', np.mean(dists[:, i]))
    #logger.debug('Average pixel distance: %f', np.mean(dists_pixel[:, i]))
    logger.debug('5cm accuracy: %f', np.sum(dists[:, i] < 5) / float(dists.shape[0]))
    logger.debug('10cm accuracy: %f', np.sum(dists[:, i] < 10) / float(dists.shape[0]))
    logger.debug('15cm accuracy: %f', np.sum(dists[:, i] < 15) / float(dists.shape[0]))
    mAP += np.sum(dists[:, i] < 10) / float(dists.shape[0])
logger.debug('mAP (10cm): %f', mAP / NUM_JOINTS)

###############################################################################
# Visualize predictions
###############################################################################

if args.make_png:
    # mkdir(outDir+dataDir+'/png/')
    for test_idx in range(num_test):
        png_path = os.path.join(args.png_dir, str(test_idx)+'.png')
        drawPred(X_test[test_idx], y_pred[test_idx], qms[test_idx], body_centers_test[test_idx], png_path, NUM_JOINTS, JOINT_NAMES)
