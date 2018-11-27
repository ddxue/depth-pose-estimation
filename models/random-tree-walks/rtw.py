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
        processed/
            CAD-60/
                depth_images.npy
                joints.npy
            ...
    models/
        random-tree-walks/
            rtw.py
            helper.py
        ...
    output/
        random-tree-walks/
            models/
            preds/
            png/
"""

###############################################################################
# Parser arguments
###############################################################################

parser = argparse.ArgumentParser(description='Random Tree Walks algorithm.')

# Loading options for the model and data
# parser.add_argument('--load-params', action='store_true',
#                     help='Load the parameters')
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
parser.add_argument('--preds-dir', type=str, default='../../output/random-tree-walks/preds',
                    help='Directory to save predictions')
parser.add_argument('--png-dir', type=str, default='../../output/random-tree-walks/png',
                    help='Directory to save prediction images')

# Training options
parser.add_argument('--seed', type=int, default=1111,
                    help='Random seed')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Shuffle the data')
parser.add_argument('--multithread', action='store_true',
                    help='Train each joint on a separate threads')

# Evaluation hyperparameters
parser.add_argument('--num-steps', type=int, default=300,
                    help='Number of steps during evaluation')
parser.add_argument('--step-size', type=int, default=2,
                    help='Step size (in cm) during evaluation')

# Output options
parser.add_argument('--make-png', action='store_true',
                    help='Draw predictions on top of inputs')

args = parser.parse_args()

###############################################################################
# Training hyperparameters
###############################################################################

# Train-test ratio
TRAIN_RATIO = 0.8

# Dimension of each feature vector
NUM_FEATS = 500
MAX_FEAT_OFFSET = 150

# Number of samples for each joint for each example
NUM_SAMPLES = 500

# Set maximum XYZ offset from each joint
MAX_XY_OFFSET = 10 # image xy coordinates (pixels)
MAX_Z_OFFSET = 0.5 # z-depth coordinates (meters)

# Number of clusters for K-Means regression
K = 20

###############################################################################
# Constants
###############################################################################

# Number of joints in a skeleton
NUM_JOINTS = 15

# List of joint names
JOINT_NAMES = ['NECK (0)', 'HEAD (1)', \
                'LEFT SHOULDER (2)', 'LEFT ELBOW (3)', 'LEFT HAND (4)', \
                'RIGHT SHOULDER (5)', 'RIGHT ELBOW (6)', 'RIGHT HAND (7)', \
                'LEFT KNEE (8)', 'LEFT FOOT (9)', \
                'RIGHT KNEE (10)', 'RIGHT FOOT (11)', \
                'LEFT HIP (12)', \
                'RIGHT HIP (13)', \
                'TORSO (14)']

# Depth image dimension
H, W = 240, 320

# See https://help.autodesk.com/view/MOBPRO/2018/ENU/?guid=__cpp_ref__nui_image_camera_8h_source_html
C = 3.8605e-3 # NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS

# Set the kinematic tree (starting from torso body center)
kinem_order =  [14,  0, 13, 12, 1, 2, 5, 3, 6, 4, 7,  8, 10, 9, 11]
kinem_parent = [-1, 14, 14, 14, 0, 0, 0, 2, 5, 3, 6, 12, 13, 8, 10]

###############################################################################
# Load dataset splits
###############################################################################

def load_dataset(processed_dir, is_mask=False):
    """Loads the depth images and joints from the processed dataset.

    Note that each joint is a coordinate of the form (im_x, im_y, depth_z).
    Each depth image is an H x W image containing depth_z values.

    depth_z values are in meters.

    @return:
        depth_images : depth images (N x H x W)
        joints : joint positions (N x NUM_JOINTS x 3)
    """
    logger.debug('Loading data from directory %s', processed_dir)

    # Load input and labels from numpy files
    depth_images = np.load(os.path.join(processed_dir, 'depth_images.npy')) # N x H x W depth images
    joints = np.load(os.path.join(processed_dir, 'joints.npy')) # N x NUM_JOINTS x 3 joint locations

    assert depth_images.shape[1] == H and depth_images.shape[2] == W, "Invalid depth image dimensions."

    # Load and apply mask to the depth images
    if is_mask:
        depth_mask = np.load(os.path.join(processed_dir, 'depth_mask.npy')) # N x H x W depth mask
        depth_images = depth_images * depth_mask

    logger.debug('Data loaded: # data: %d', depth_images.shape[0])
    return depth_images[:2000], joints[:2000]

def compute_params(joints, num_feats=NUM_FEATS, max_feat_offset=MAX_FEAT_OFFSET):
    """Computes the body centers for each skeleton.

    @params:
        max_feat_offset : the maximum offset for features (before divided by d)
        num_feats : the number of features of each offset point
    """
    logger.debug('Computing parameters...')

    # Calculate the body centers by weighted average of joints
    left_hips = (joints[:,2] + 2 * joints[:,8]) / 3.0
    right_hips = (joints[:,5] + 2 * joints[:,10]) / 3.0
    body_centers = (joints[:,2] + left_hips + joints[:,5] + right_hips) / 4.0

    # Compute the theta = (-max_feat_offset, max_feat_offset) for 4 coordinates (x1, x2, y1, y2)
    theta = np.random.randint(-max_feat_offset, max_feat_offset + 1, (4, num_feats)) # (4, num_feats)

    return body_centers, theta

def split_dataset(X, y, body_centers, train_ratio):
    """Splits the dataset according to the train-test ratio.

    @params:
        X : depth images (N x H x W)
        y : joint positions (N x NUM_JOINTS x 3)
        train_ratio : ratio of training to test
    """
    test_ratio = 1.0 - train_ratio
    num_test = int(X.shape[0] * test_ratio)

    X_train, y_train = X[num_test:], y[num_test:]
    X_test, y_test = X[:num_test], y[:num_test]
    body_centers_train, body_centers_test = body_centers[num_test:], body_centers[:num_test]

    logger.debug('Data split: # training data: %d, # test data: %d', X_train.shape[0], X_test.shape[0])
    return X_train, y_train, X_test, y_test, body_centers_train, body_centers_test

processed_dir = os.path.join(args.input_dir, args.dataset) # directory of saved numpy files

depth_images, joints = load_dataset(processed_dir)
body_centers, theta = compute_params(joints)
X_train, y_train, X_test, y_test, body_centers_train, body_centers_test = split_dataset(depth_images, joints, body_centers, TRAIN_RATIO)

num_train = X_train.shape[0]
num_test = X_test.shape[0]

###############################################################################
# Train model
###############################################################################

def get_features(img, q, z, theta):
    """Gets the feature vector for a single example.

    @params:
        img : depth image = (H x W)
        q : joint xyz position with some random offset vector
        z : z-value of body center
        theta : (-max_feat_offset, max_feat_offset) = (4, num_feats)
    """
    # Retrieve the (y, x) of the joint offset coordinates
    coor = q[:2][::-1] # coor: flip x, y -> y, x
    coor[0] = np.clip(coor[0], 0, H-1) # limits y between 0 and H
    coor[1] = np.clip(coor[1], 0, W-1) # limits x between 0 and W
    coor = np.rint(coor).astype(int) # rounds to nearest integer

    # Find z-value of joint offset by indexing into depth imag
    LARGE_NUM = 100
    img[img == 0] = LARGE_NUM # no division by zero
    dq = z if (img[tuple(coor)] == LARGE_NUM) else img[tuple(coor)] # initialize to LARGE_NUM

    # Normalize x theta by z-value
    x1 = np.clip(coor[1] + theta[0] / dq, 0, W-1).astype(int)
    x2 = np.clip(coor[1] + theta[2] / dq, 0, W-1).astype(int)

    # Normalize y theta by z-value
    y1 = np.clip(coor[0] + theta[1] / dq, 0, H-1).astype(int)
    y2 = np.clip(coor[0] + theta[3] / dq, 0, H-1).astype(int)

    # Get the feature vector as difference of depth-values
    feature = img[y1, x1] - img[y2, x2]
    return feature

def get_random_offset(max_offset_xy=MAX_XY_OFFSET, max_offset_z=MAX_Z_OFFSET):
    """Gets xyz vector with uniformly random xy and z offsets.
    """
    offset_xy = np.random.randint(-max_offset_xy, max_offset_xy + 1, 2)
    offset_z = np.random.uniform(-max_offset_z, max_offset_z, 1)
    offset = np.concatenate((offset_xy, offset_z)) # xyz offset
    return offset

def get_training_samples(joint_id, X, y, body_centers, theta, num_feats=NUM_FEATS, num_samples=NUM_SAMPLES):
    """Generates training samples for each joint.

    Each sample is (i, q, u, f) where:
         i is the index of the depth image,
         q is the random offset point from the joint,
         u is the unit direction vector toward the joint location,
         f is the feature array

    @params:
        X : depth images (N x H x W)
        y : joint positions (im_x, im_y, depth_z)
        joint_id : current joint id
        num_samples : number of samples of each joint
        max_offset_xy : maximum offset for samples in (x, y) axes
        max_offset_z : maximum offset for samples in z axis

    @return:
        S_f : samples feature array (N x num_samples x num_feats)
        S_u : samples unit direction vectors (N x num_samples x 3)
    """
    num_train, _, _ = X.shape

    S_f = np.zeros((num_train, num_samples, num_feats), dtype=np.float64)
    S_u = np.zeros((num_train, num_samples, 3), dtype=np.float64)

    for train_idx in range(num_train):
        if train_idx % 100 == 0:
            logger.debug('Joint %s: Processing image %d / %d', JOINT_NAMES[joint_id], train_idx, num_train)

        # Create samples for each training example
        for sample_idx in range(num_samples):
            depth_im = X[train_idx]
            offset = get_random_offset()
            unit_offset = 0 if np.linalg.norm(offset) == 0 else (-offset / np.linalg.norm(offset))
            body_center_z = body_centers[train_idx][2]

            S_f[train_idx, sample_idx] = get_features(depth_im, y[train_idx] + offset, body_center_z, theta)
            S_u[train_idx, sample_idx] = unit_offset

    return S_f, S_u

def stochastic(regressor, features, unit_directions):
    """Applies stochastic relaxation when choosing the unit direction. Training
    samples at the leaf nodes are further clustered using K-means.
    """
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

def train(joint_id, X, y, model_dir, min_samples_leaf=400, load_models=args.load_model):
    """Trains a regressor tree on the unit directions towards the joint.

    @params:
        joint_id : current joint id
        X : samples feature array (N x num_samples x num_feats)
        y : samples unit direction vectors (N x num_samples x 3)
        min_samples_split : minimum number of samples required to split an internal node
        load_models : load trained models from disk (if exist)
    """
    logger.debug('Start training %s model...', JOINT_NAMES[joint_id])

    regressor_path = os.path.join(model_dir, 'regressor' + str(joint_id) + '.pkl')
    L_path = os.path.join(model_dir, 'L' + str(joint_id) + '.pkl')

    # Load saved model from disk
    if load_models and (os.path.isfile(regressor_path) and os.path.isfile(L_path)):
        logger.debug('Loading model %s from files...', JOINT_NAMES[joint_id])

        regressor = pickle.load(open(regressor_path, 'rb'))
        L = pickle.load(open(L_path, 'rb'))
        return regressor, L

    X_reshape = X.reshape(X.shape[0] * X.shape[1], X.shape[2]) # (N x num_samples, num_feats)
    y_reshape = y.reshape(y.shape[0] * y.shape[1], y.shape[2]) # (N x num_samples, 3)

    # Count the number of valid (non-zero) samples
    valid_rows = np.logical_not(np.all(X_reshape == 0, axis=1)) # inverse of invalid samples
    logger.debug('Model %s - Valid samples: %d / %d', JOINT_NAMES[joint_id], X_reshape[valid_rows].shape[0], X_reshape.shape[0])

    # Fit decision tree to samples
    regressor = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf)
    regressor.fit(X_reshape[valid_rows], y_reshape[valid_rows])

    L = stochastic(regressor, X_reshape, y_reshape)

    # Print statistics on leafs
    leaf_ids = regressor.apply(X_reshape)
    bin = np.bincount(leaf_ids)
    unique_ids = np.unique(leaf_ids)
    biggest = np.argmax(bin)
    smallest = np.argmin(bin[bin != 0])

    logger.debug('Model %s - # Leaves: %d', JOINT_NAMES[joint_id], unique_ids.shape[0])
    logger.debug('Model %s - Smallest Leaf ID: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], smallest, bin[bin != 0][smallest], np.sum(bin))
    logger.debug('Model %s - Biggest Leaf ID: %d, # Samples: %d/%d', JOINT_NAMES[joint_id], biggest, bin[biggest], np.sum(bin))
    logger.debug('Model %s - Average Leaf Size: %d', JOINT_NAMES[joint_id], np.sum(bin) / unique_ids.shape[0])

    # Save models to disk
    pickle.dump(regressor, open(regressor_path, 'wb'))
    pickle.dump(L, open(L_path, 'wb'))

    return regressor, L

def train_parallel(joint_id, X, y, body_centers, theta, model_dir, regressor_queue, L_queue):
    """Train each join in parallel.
    """
    S_f, S_u = get_training_samples(joint_id, X, y, body_centers, theta)
    regressor, L = train(joint_id, S_f, S_u, model_dir)
    regressor_queue.put({joint_id: regressor})
    L_queue.put({joint_id: L})

def train_series(joint_id, X, y, body_centers, theta, model_dir):
    """Train each joint sequentially.
    """
    S_f, S_u = get_training_samples(joint_id, X, y, body_centers, theta)
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

    regressors = dict(list(i.items())[0] for i in regressors_tmp)
    Ls = dict(list(i.items())[0] for i in Ls_tmp)

    [p.join() for p in processes]

###############################################################################
# Evaluate model
###############################################################################

def test_model(regressor, L, theta, qm0, img, body_center, num_steps=args.num_steps, step_size=args.step_size):
    qm = np.zeros((num_steps + 1, 3))
    qm[0] = qm0
    joint_pred = np.zeros(3)

    for i in range(num_steps):
        body_center_z = body_center[2]
        f = get_features(img, qm[i], body_center_z, theta).reshape(1, -1) # flatten feature vector
        leaf_id = regressor.apply(f)[0]

        idx = np.random.choice(K, p=L[leaf_id][0]) # L[leaf_id][0] = weights
        u = L[leaf_id][1][idx] # L[leaf_id][1] = centers

        qm[i+1] = qm[i] + u * step_size
        qm[i+1][0] = np.clip(qm[i+1][0], 0, W-1) # limit x between 0 and W
        qm[i+1][1] = np.clip(qm[i+1][1], 0, H-1) # limit y between 0 and H
        qm[i+1][2] = img[int(qm[i+1][1]), int(qm[i+1][0])] # index (y, x) into image for z position
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
        qms[test_idx][joint_id], y_pred[test_idx][joint_id] = test_model(regressors[joint_id], Ls[joint_id], theta, qm0, X_test[test_idx], body_centers_test[test_idx])
        local_error[test_idx, :, joint_id, :] = y_test[test_idx, joint_id] - qms[test_idx][joint_id]

y_pred[:, :, 2] = y_test[:, :, 2]

# np.save(modelsDir + 'qms.npy', qms)
# np.save(modelsDir + 'y_pred.npy', y_pred)
# np.save(modelsDir + 'local_error.npy', local_error)
#
# for joint_id in range(NUM_JOINTS):
#     # print(y_test[:, joint_id].shape)
#     np.savetxt(outDir+modelsDir+'/pred/'+JOINT_NAMES[joint_id]+'_test.txt', y_test[:, joint_id], fmt='%.3f')
#     # print(y_pred[:, jointID].shape)
#     np.savetxt(outDir+modelsDir+'/pred/'+JOINT_NAMES[joint_id]+'_pred.txt', y_pred[:, joint_id], fmt='%.3f ')

###############################################################################
# Run evaluation metrics
###############################################################################

logger.debug('\n------- Computing evaluation metrics -------')

def get_distances(y_test, y_pred):
    """Compute the raw world distances between the prediction and actual joint
    locations.
    """
    assert y_test.shape == y_pred.shape, "Mismatch of y_test and y_pred"

    distances = np.zeros((y_test.shape[:2]))
    for i in range(y_test.shape[0]):
        p1 = pixel2world(y_test[i], C)
        p2 = pixel2world(y_pred[i], C)
        distances[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))
    return distances

distances = get_distances(y_test, y_pred) * 100.0 # convert from m to cm

distances_path = os.path.join(args.preds_dir, 'distances.txt')
np.savetxt(distances_path, distances, fmt='%.3f')

distances_pixel = np.zeros((y_test.shape[:2]))
for i in range(y_test.shape[0]):
   p1 = y_test[i]
   p2 = y_pred[i]
   distances_pixel[i] = np.sqrt(np.sum((p1-p2)**2, axis=1))

mAP = 0
for i in range(NUM_JOINTS):
    logger.debug('\nJoint %s:', JOINT_NAMES[i])
    logger.debug('Average distance: %f cm', np.mean(distances[:, i]))
    logger.debug('Average pixel distance: %f', np.mean(distances_pixel[:, i]))
    logger.debug('5cm accuracy: %f', np.sum(distances[:, i] < 5) / float(distances.shape[0]))
    logger.debug('10cm accuracy: %f', np.sum(distances[:, i] < 10) / float(distances.shape[0]))
    logger.debug('15cm accuracy: %f', np.sum(distances[:, i] < 15) / float(distances.shape[0]))
    mAP += np.sum(distances[:, i] < 10) / float(distances.shape[0])

logger.debug('mAP (10cm): %f', mAP / NUM_JOINTS)

###############################################################################
# Visualize predictions
###############################################################################

# if args.make_png:
logger.debug('\n------- Saving prediction visualizations -------')

for test_idx in range(num_test):
    png_path = os.path.join(args.png_dir, str(test_idx) + '.png')
    drawPred(X_test[test_idx], y_pred[test_idx], qms[test_idx], body_centers_test[test_idx], png_path, NUM_JOINTS, JOINT_NAMES)
