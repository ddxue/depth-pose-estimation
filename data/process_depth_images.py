import numpy as np
from scipy.spatial import distance
import glob
import os
import cv2
import sys

W = 320
H = 240
nJoints = 12

C = 3.8605e-3 #NUI_CAMERA_DEPTH_NOMINAL_INVERSE_FOCAL_LENGTH_IN_PIXELS
np.set_printoptions(threshold=np.nan)

jointNameEVAL = ['NECK', 'HEAD', 'LEFT SHOULDER', 'LEFT ELBOW', \
                'LEFT HAND', 'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT HAND', \
                'LEFT KNEE', 'LEFT FOOT', 'RIGHT KNEE', 'RIGHT FOOT', \
                'LEFT HIP', 'RIGHT HIP', 'TORSO']

skeleton = [(0,1), (0,2), (2,3), (3,4), (0,5), (5,6), (6,7), (14,8), \
            (8, 9), (14,10), (10,11), (14,2), (14,5), (14,12), (14,13)]

palette = [(34, 88, 226), (34, 69, 101), (0, 195, 243), (146, 86, 135), \
           (38, 61, 43), (241, 202, 161), (50, 0, 190), (128, 178, 194), \
           (23, 45, 136), (0, 211, 220), (172, 143, 230), (108, 68, 179), \
           (121, 147, 249), (151, 78, 96), (0, 166, 246), (165, 103, 0), \
           (86, 136, 0), (130, 132, 132), (0, 182, 141), (0, 132, 243)] # BGR

def visualizeImgAndJoints(imgs, joints, path, show=False, write=False):
    for i in range(imgs.shape[0]):
        print 'average depth: %f' % np.mean(imgs[i][imgs[i] != 0])
        img = (imgs[i]-np.amin(imgs[i]))*255.0/(np.amax(imgs[i])-np.amin(imgs[i]))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        imgSum = np.sum(img, 2)
        img[imgSum == 0] = [0, 0, 255]

        for j in range(joints.shape[1]):
            cv2.circle(img, tuple(joints[i, j, :2].astype(np.uint16)), \
                2, (255, 0, 0), -1)

        if show:
            cv2.imshow('img', img)
            cv2.waitKey(0)
        if write:
            cv2.imwrite(path+'/img'+str(i)+'.jpg', img)
    cv2.destroyAllWindows()

def pixel2world(pixel, C):
    world = np.empty(pixel.shape)
    world[:, 2] = pixel[:, 2]
    world[:, 0] = (pixel[:, 0]-W/2.0)*C*pixel[:, 2]
    world[:, 1] = -(pixel[:, 1]-H/2.0)*C*pixel[:, 2]
    return world

def world2pixel(world, C):
    pixel = np.empty(world.shape)
    pixel[:, 2] = world[:, 2]
    pixel[:, 0] = np.rint(world[:, 0]/world[:, 2]/C + W/2.0)
    pixel[:, 1] = np.rint(-world[:, 1]/world[:, 2]/C + H/2.0)
    return pixel

def x2yz(pt1, pt2, x):
    y = (x-pt1[0])/(pt2[0]-pt1[0])*(pt2[1]-pt1[1])+pt1[1]
    z = (x-pt1[0])/(pt2[0]-pt1[0])*(pt2[2]-pt1[2])+pt1[2]
    return (y,z)

def y2xz(pt1, pt2, y):
    x = (y-pt1[1])/(pt2[1]-pt1[1])*(pt2[0]-pt1[0])+pt1[0]
    z = (y-pt1[1])/(pt2[1]-pt1[1])*(pt2[2]-pt1[2])+pt1[2]
    return (x,z)

def joints2skeleton(joints):
    ptsAll = np.array([]).reshape(0, 3)
    ptsPerJoint = 5.0
    leftHip = (joints[2]+2*joints[8])/3.0
    rightHip = (joints[5]+2*joints[10])/3.0
    torso = (joints[2]+leftHip+joints[5]+rightHip)/4.0
    joints = np.vstack((joints, leftHip, rightHip, torso))

    for i, jointPair in enumerate(skeleton):
        x, y, z, nPts = None, None, None, 0
        pt1 = joints[jointPair[0]]
        pt2 = joints[jointPair[1]]

        idx, nPts = 0, 0
        if abs(pt1[0]-pt2[0]) < abs(pt1[1]-pt2[1]):
            idx = 1

        start = min(pt1[idx], pt2[idx])
        end = max(pt1[idx], pt2[idx])
        if idx == 1:
            y = np.arange(start, end, (end-start)/ptsPerJoint)
            x, z = y2xz(pt1, pt2, y)
            nPts = y.shape[0]
        else:
            x = np.arange(start, end, (end-start)/ptsPerJoint)
            y, z = x2yz(pt1, pt2, x)
            nPts = x.shape[0]
        if nPts == 0:
            continue
        pts = np.vstack((x, y, z)).T
        ptsAll = np.concatenate((ptsAll, pts))

    # torso
    tl = joints[2]
    tr = joints[5]
    bl = joints[12]
    br = joints[13]

    torsoSkel = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
    for i, jointPair in enumerate(torsoSkel):
        x, y, z, nPts = None, None, None, 0
        pt1 = torsoSkel[i][0]
        pt2 = torsoSkel[i][1]

        idx, nPts = 0, 0
        if abs(pt1[0]-pt2[0]) < abs(pt1[1]-pt2[1]):
            idx = 1

        start = min(pt1[idx], pt2[idx])
        end = max(pt1[idx], pt2[idx])
        if idx == 1:
            y = np.arange(start, end, (end-start)/ptsPerJoint)
            x, z = y2xz(pt1, pt2, y)
            nPts = y.shape[0]
        else:
            x = np.arange(start, end, (end-start)/ptsPerJoint)
            y, z = x2yz(pt1, pt2, x)
            nPts = x.shape[0]
        if nPts == 0:
            continue

        pts = np.vstack((x, y, z)).T
        ptsAll = np.concatenate((ptsAll, pts))

    return ptsAll

def bgSub(img, joints):
    threshold = 0.2
    dists = distance.cdist(img, joints, 'euclidean')
    dists = np.amin(dists, axis=1)

    # a = joints2skeleton(joints)
    # print a.shape
    # assert False
    return dists < threshold

def main():
    getJpg, getJoints, getNpArray = False, False, False
    dataDir = '/mnt0/data/EVAL/data'
    datasets = glob.glob(dataDir+'/*')
    datasets.sort()
    #print datasets

    jpgDir = dataDir + '/jpg'
    if not os.path.exists(jpgDir):
        os.makedirs(jpgDir)

    getJpg = True
    getJoints = True
    getNpArray = True

    joints = []
    skeletons = []
    imgs = []
    imgs_mask = []

    for dataset in datasets:
        inImgsDir = dataset + '/depth/'
        inJointsDir = dataset + '/joints/'

        paths = glob.glob(inJointsDir + '*.txt')
        paths.sort()
        #print paths
        for i, jointPath in enumerate(paths):
            imgPath = jointPath.replace('joints', 'depth')
            joint = np.loadtxt(jointPath)
            img = np.loadtxt(imgPath)
            if joint.shape[0] != nJoints:
                continue
            if img.shape[0] != H*W:
                continue
            indicesJoint = np.nonzero(joint[:, 2])
            indicesImg = np.nonzero(img[:, 2])
            if len(indicesJoint[0]) == 0 or len(indicesImg[0]) == 0:
                continue

            joint[:, 2] *= -1
            joint_pixel = world2pixel(joint, C)
            skeleton = joints2skeleton(joint)
            skeleton_pixel = world2pixel(skeleton, C)

            img[:, 2] *= -1
            img_pixel = world2pixel(img[indicesImg], C)
            img_np = np.zeros((H, W))
            img_np[img_pixel[:, 1].astype(int), img_pixel[:, 0].astype(int)] = img_pixel[:, 2]

            img_mask = bgSub(img, skeleton)
            img_mask_pixel = world2pixel(img[img_mask], C)
            img_mask_np = np.zeros((H, W))
            img_mask_np[img_mask_pixel[:, 1].astype(int), img_mask_pixel[:, 0].astype(int)] = 1

            joints.append(joint_pixel)
            skeletons.append(skeleton_pixel)
            imgs.append(img_np)
            imgs_mask.append(img_mask_np)

    joints = np.array(joints)
    skeletons = np.array(skeletons)
    imgs = np.array(imgs)
    imgs_mask = np.array(imgs_mask)
    #visualizeImgAndJoints(imgs_mask, skeletons)
    print joints.shape, imgs.shape, imgs_mask.shape
    visualizeImgAndJoints(imgs*imgs_mask, joints, path=jpgDir, write=True)
    np.save(dataDir+'/joints.npy', joints)
    np.save(dataDir+'/I.npy', imgs)
    np.save(dataDir+'/I_mask.npy', imgs_mask)

def main1():
    #I_mask = np.load('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/I_mask.npy')
    #I = np.load('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/I.npy')
    #np.save('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/img_mask.npy', I_mask[0])
    #np.save('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/img.npy', I[0])

    img_mask = np.load('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/img_mask.npy')
    img = np.load('/mnt0/alan/healthcare/src/poseEstimation/RTW/data_EVAL/img.npy')

    #img[img_mask != 0] = img[img_mask != 0]*0.5

    img = (img-np.amin(img))*255.0/(np.amax(img)-np.amin(img))
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img[img == 0] = np.mean(img)*2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def main2():
    idx = str(30).zfill(4)
    path = '/mnt0/data/EVAL/data/seq_a_0/'
    img = np.load(path+'nparray_depthcoor/'+'seq_a_0_'+idx+'.npy')
    joints = np.loadtxt(path+'joints_depthcoor/'+'seq_a_0_'+idx+'.txt')

    print img.shape, joints.shape

    img = (img-np.amin(img))*255.0/(np.amax(img)-np.amin(img))
    img = img.astype(np.uint8)
    img[img == 255] = np.mean(img)*2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

    for i in range(12):
        cv2.circle(img, tuple(joints[i, :2].astype(np.uint16)), 3, palette[i], -1)

    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    cv2.imwrite('/mnt0/alan/healthcare/src/poseEstimation/RTW/visualize/EVAL.jpg', img)

if __name__ == "__main__":
    main2()
