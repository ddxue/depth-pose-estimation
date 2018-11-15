import os
import json
import argparse
import numpy as np
from scipy import misc
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

NUM_JOINTS = 5
HEIGHT = 160
WIDTH = 120

# creates sparse matrix
# matrix[i, j] -> 1 iff (i, j) in labels, 0 else
def label_mask(image, labels):
	mask = np.zeros_like(image)
	labels = labels[labels[:, :, 0] < WIDTH, :]
	labels = labels[labels[:, 1] < HEIGHT, :]
	labels = labels.astype(int)
	mask[labels[:,1], labels[:,0]] = 1
	return mask

# converts .txt raw label to ndarray
# shape is (num_people, num_joints, 2)
def ndarray_convert(joints_directory):
	joint_filenames = set(os.listdir(joints_directory))
	joints_map = {}

	while joint_filenames:
		joint_filename = joint_filenames.pop()
		with open(os.path.join(joints_directory, joint_filename)) as joint_file:
			try:
				coord_pairs = joint_file.readlines()[1:]
			except:
				continue
			num_people = len(coord_pairs) / (NUM_JOINTS + 1)
			joints = np.zeros((num_people, NUM_JOINTS, 2))
			for i, coord_pair in enumerate(coord_pairs):
				if i % (NUM_JOINTS + 1) != 0:
					person = (i - 1) / (NUM_JOINTS + 1)
					joint = (i - 1) % (NUM_JOINTS + 1)
					joints[person, joint] = coord_pair.split(' ')[0:2]
			if joints.size > 0:
				joints_map[os.path.splitext(joint_filename)[0]] = joints

	return joints_map

# computes the average euclidean distance labels and predictions
def compute_euclidean_RMSE(x_labels, x_preds, y_labels, y_preds):
	labels = np.dstack([x_labels, y_labels])[0]
	preds = np.dstack([x_preds, y_preds])[0]
	x_squared = (labels[:,0] - preds[:,0]) ** 2
	y_squared = (labels[:,1] - preds[:,1]) ** 2
	distances = np.sqrt(x_squared + y_squared)
	return np.sum(distances, dtype='float64') / distances.size

def baseline_train(images_directory, joints_directory, training):
	image_filenames = set(os.listdir(images_directory))
	joints_map = ndarray_convert(joints_directory)
	filenames = []
	image_x = []
	label_x = []
	label_y = []

	while image_filenames:
		image_filename = image_filenames.pop()
		filename, extension = os.path.splitext(image_filename)
		if extension.lower() != '.png': continue
		if filename in joints_map:
			filenames.append(filename)
			image = misc.imread(os.path.join(images_directory, image_filename))
			image_x.append(image.flatten())
			labels = joints_map[filename][0, 0]
			label_x.append(labels[0])
			label_y.append(labels[1])

	filenames = np.array(filenames)
	label_x = np.array(label_x)
	label_y = np.array(label_y)
	num_samples = len(image_x)

	if training:
		training_size = len(image_x)

		X_train = np.vstack(image_x[:training_size])
		xlabel_train = label_x[:training_size]
		ylabel_train = label_y[:training_size]

		# X_dev = np.vstack(image_x[training_size:])
		# dev_xlabel = label_x[training_size:]
		# dev_ylabel = label_y[training_size:]

		clf_xneck = svm.SVC()
		clf_xneck.fit(X_train, xlabel_train)
		joblib.dump(clf_xneck, 'clf_xneck.pkl')

		clf_yneck = svm.SVC()
		clf_yneck.fit(X_train, ylabel_train)
		joblib.dump(clf_yneck, 'clf_yneck.pkl')

		# dev_xpreds = clf_xneck.predict(X_dev)
		# dev_ypreds = clf_xneck.predict(X_dev)

		# x_error = mean_squared_error(dev_xpreds, dev_xlabel) ** 0.5
		# y_error = mean_squared_error(dev_ypreds, dev_ylabel) ** 0.5
		# print "Dev RMSE on X: %s" % (x_error)
		# print "Dev RMSE on Y: %s" % (y_error)
		# print "Dev Average Euclidean Distance: %s" % (compute_euclidean_RMSE(dev_xlabel, dev_xpreds, dev_ylabel, dev_ypreds))
		# return np.dstack((filenames, dev_xpreds, dev_ypreds))[0]

	else:
		X_test = np.vstack(image_x)
		test_xlabel = label_x
		test_ylabel = label_y

		clf_xneck = joblib.load('clf_xneck.pkl') 
		clf_yneck = joblib.load('clf_yneck.pkl')

		test_xpreds = np.random.randint(0, 120, size=len(image_x)) #  clf_xneck.predict(X_test)
		test_ypreds = np.random.randint(0, 160, size=len(image_x)) #  clf_yneck.predict(X_test)

		x_error = mean_squared_error(test_xpreds, test_xlabel) ** 0.5
		y_error = mean_squared_error(test_ypreds, test_ylabel) ** 0.5
		print "Test RMSE on X: %s" % (x_error)
		print "Test RMSE on Y: %s" % (y_error)
		print "Test Average Euclidean Distance: %s" % (compute_euclidean_RMSE(test_xlabel, test_xpreds, test_ylabel, test_ypreds))
		np.save("./test_preds", np.dstack((filenames, test_xpreds, test_ypreds))[0])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='run baseline svm on image data and joint labels')
	parser.add_argument('--images', required=True, help='path to directory containing image data (.png)')
	parser.add_argument('--joints', required=True, help='path to directory containing coordinate data (.npy)')
	parser.add_argument('--training', default=False, action='store_true', help='whether the model is being retrained')
	args = parser.parse_args()
	baseline_train(args.images, args.joints, args.training)
