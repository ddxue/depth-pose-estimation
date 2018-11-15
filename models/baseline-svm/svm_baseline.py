from scipy import misc
from sklearn import svm
import numpy as np
import os

dataDir = '/scail/data/group/vision/u/syyeung/hospital/data/' 
dataset = 'cvpr10-19-15morning'
labelPath = ''
imageType = 'rgb'

def readData(fileDir, cropX, cropY, cropWidth, cropHeight):
  files = [f for f in os.listdir(fileDir) if f.endswith('jpg')]
  imgData = np.zeros((files.size, cropWidth*cropHeight))
  for f in files
    im = misc.imread(f);
    imCrop = im[cropY:cropY+cropHeight, cropX:cropX+cropWidth]
    imgData[i,:] = np.reshape(imCrop,(1 , imCrop.size))
  
  return imgData

def readLabel(fileName):
  label = np.loadtxt(fileName)
  return label

def processData(data,label):


def train(data, label):
  clf = svm.SVC()
  clf.fit(data,label)

  return clf

def test(data, clf):
  return clf.predict(data)

def main():
 imgData = readData(dataDir, 114,240,64,64)
 imgLabel = readLabel(labelPath)
 [trainData testData trainLabel testLabel] = processData(imgData,imgLabel)
 clf = train(trainData,trainLabel);
 prediction = test(testData,clf);


if __name__ == "__main__":
      main()
