import matplotlib.pyplot as plt
import numpy as np

out_path = './out/'
out_images = out_path + 'images.npy'
train_ratio = 0.7

def main():
  images = np.load(out_images)
  num_train = round(0.7 * images.shape[0])
  x = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]) * num_train
  y = np.array([0.326565291108, 0.32614352107, 0.343795418489, 0.36116978344, 0.354532922755, 0.371552355023])
  plt.scatter(x, y)
  plt.plot(x, y)
  plt.xlabel('Num. training images')
  plt.ylabel('Average per-class accuracy')
  plt.savefig('rf.jpg')

if __name__ == "__main__":
  main()

