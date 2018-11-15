import cv2
import numpy as np
import os
from os import listdir

def bwareaopen(img, area):
    img2 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)
    maxIdx = -1
    maxArea = 0
    for i in np.arange(len(contours)):
        curArea = cv2.contourArea(contours[i])
        if (curArea >= area and curArea >= maxArea):
        	maxIdx = i
        	maxArea = curArea

    if maxIdx != -1:
    	cv2.drawContours(mask, contours, maxIdx, 255, -1)
    	img = cv2.bitwise_and(img, mask)
    	#img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #cv2.drawContours(img, contours, maxIdx, (0, 0, 255), 2)
        #cv2.drawContours(img, contours, maxIdx, 255, 2)
    	return img
    else:
    	return None

def removeZeros(img, mean, lowThred):
	rows = img.shape[0]
	cols = img.shape[1]
	for i in xrange(rows):
		for j in xrange(cols):
			k = img.item(i, j)
			if k <= lowThred:
				img.itemset((i, j), mean[i])
	return img

def createFeatures(img, mask):
	n = np.count_nonzero(mask)
	rows, cols = np.nonzero(mask)
	depth = img[rows, cols]

	features = np.dstack((rows, cols, depth))[0]
	#features[:, 0] = rows
	#features[:, 1] = cols
	       
def goodExample(img):
	if img == None:
		return False
	sum = 0

	sums = np.sum(img, axis=0)
	for i in range(0, 10) + range(len(sums) - 10, len(sums)):
		sum = sum + sums[i]

	sums = np.sum(img, axis=1)
	for i in range(0, 10):
		sum = sum + sums[i]

	if sum > 0:
		return False
	else:
		return True

# parameters
imgDir = '/Users/alan/Documents/research/dataset/new/'
dataSets = {'cvpr10-18-15morning', 'cvpr10-19-15morning'}
rename = False # rename in sequential
ext = '.jpg'
lowThred = 3 # solving the blackhole issue
bgRatio = 0.5 # percentage of images used to calculate the background
meanRatio = 0.7 # percentage of pixels used in each row to calculate the mean
# end parameters

outDir = imgDir + 'out/'
if not os.path.exists(outDir):
	os.makedirs(outDir)

imgFiles = []
for dataSet in dataSets:
	dataSetsDir = imgDir + dataSet + '/d/'
	files = [f for f in listdir(dataSetsDir) if f.endswith(ext)]
	files = [dataSet + '/d/' + i for i in files]
	imgFiles = np.append(imgFiles, files)

#imgFiles.sort(key=lambda x: int(x.split('-')[1][:-len(ext)]))
#print(imgFiles)
tmp = cv2.imread(imgDir + imgFiles[0], 0)
sz = tmp.shape

# calculate mean of each row, and sum of each image
sums = np.zeros(len(imgFiles))
mean = np.zeros(sz[0])
for i, imgFile in enumerate(imgFiles):
	img = cv2.imread(imgDir + imgFile, 0)
	sums[i] = np.sum(img)
	img = np.fliplr(np.sort(img))
	img = img[:,0:sz[1]*meanRatio]
	mean = mean + np.mean(img, axis=1)
indices = np.argsort(sums)[::-1][:len(imgFiles)*bgRatio]
mean = mean/len(imgFiles)

# calculate the background
bg = np.zeros(sz)
for idx in indices:
	img = cv2.imread(imgDir + imgFiles[idx], 0)
	img = removeZeros(img, mean, lowThred)
	bg = bg + img.astype(np.float)
bg = bg/len(indices)
bg = bg.astype(np.uint8)
#print(np.amin(bg), np.amax(bg))
cv2.imwrite(outDir + 'bg.jpg', bg, [cv2.IMWRITE_JPEG_QUALITY, 100])

i = 0
for imgFile in imgFiles:
	img = cv2.imread(imgDir + imgFile, 0)
	img = removeZeros(img, mean, lowThred)
	img = cv2.subtract(bg, img)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	#img = cv2.erode(img, element)
	#img = cv2.dilate(img, element)
	#img = cv2.fastNlMeansDenoising(img, None, 5)
	img = bwareaopen(img, 2000);
	if not goodExample(img):
		continue
	#createFeatures(img, mask)
	img = cv2.equalizeHist(img)

	#cv2.imshow('img', img)
	#cv2.waitKey()
	if rename:
		cv2.imwrite(outDir + str(i) + ext, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
	else:
		fileName = imgFile.replace('/d/', '-')
		cv2.imwrite(outDir + fileName, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
	i = i + 1
