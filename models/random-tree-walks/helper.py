import glob
import numpy as np
import cv2
import os
import logging
import os.path
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(threshold=np.nan)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# H = 240
# W = 320
H, W = 424, 512


palette = [(34, 88, 226), (34, 69, 101), (0, 195, 243), (146, 86, 135), \
           (38, 61, 43), (241, 202, 161), (50, 0, 190), (128, 178, 194), \
           (23, 45, 136), (0, 211, 220), (172, 143, 230), (108, 68, 179), \
           (121, 147, 249), (151, 78, 96), (0, 166, 246), (165, 103, 0), \
           (86, 136, 0), (130, 132, 132), (0, 182, 141), (0, 132, 243)] # BGR

jointNameEVAL = ['NECK', 'HEAD', 'LEFT SHOULDER', 'LEFT ELBOW', \
                'LEFT HAND', 'RIGHT SHOULDER', 'RIGHT ELBOW', 'RIGHT HAND', \
                'LEFT KNEE', 'LEFT FOOT', 'RIGHT KNEE', 'RIGHT FOOT', \
                'LEFT HIP', 'RIGHT HIP', 'TORSO']
jointNameITOP = ['HEAD', 'NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', \
                'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_HAND', 'RIGHT_HAND', \
                'TORSO', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', \
                'RIGHT_KNEE', 'LEFT_FOOT', 'RIGHT_FOOT']

trainTestITOP = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # train = 0, test = 1
kinemOrderEVAL =   [0, 1, 2, 5, 3, 6, 4, 7, 8, 10, 9, 11]
kinemParentEVAL = [-1, 0, 0, 0, 2, 5, 3, 6, -1, -1, 8, 10]
kinemOrderITOP =   [8, 1, 0, 9, 10, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14]
kinemOrderITOPUpper = [8, 1, 0, 2, 3, 4, 5, 6, 7]
kinemParentITOP = [-1, 8, 1, 8, 8,  1, 1, 2, 3, 4, 5, 9,  10, 11, 12]

def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != 17:
            raise
        pass

def getImgsAndJointsITOP(dataDir, nJoints, isTop=False, maxN=None):
    global trainTestITOP
    I_train, I_test = np.empty((0, H, W), np.float16), \
        np.empty((0, H, W), np.float16)
    joints_train, joints_test = np.empty((0, nJoints, 3)), \
        np.empty((0, nJoints, 3))

    if maxN is not None:
        trainTestITOP = [1, 0]

    fileName = 'top.npy' if isTop else 'side.npy'
    for i, isTest in enumerate(trainTestITOP):
        depthPath = dataDir + '/' + str(i).zfill(2) + '_depth_' + fileName
        jointsPath = dataDir + '/' + str(i).zfill(2) + '_joints_' + fileName
        maskPath = dataDir + '/' + str(i).zfill(2) + '_predicts_' + fileName
        logger.debug('loading %s', depthPath)
        I = np.load(depthPath)
        joints = np.load(jointsPath)[:, :, :3]
        #print(type(depthPath[0,0,0]))
        I /= 1000.0
        joints[:, :, 2] /= 1000.0
        mask = np.load(maskPath)
        mask[mask >= 0] = 1
        mask[mask == -1] = 0
        I *= mask

        if isTest == 0:
            I_train = np.append(I_train, I, axis=0).astype('float16')
            joints_train = np.append(joints_train, joints, axis=0)
        else:
            I_test = np.append(I_test, I, axis=0).astype('float16')
            joints_test = np.append(joints_test, joints, axis=0)

    if maxN is not None:
        I_train = I_train[:maxN]
        I_test = I_test[:(maxN/10)]
        joints_train = joints_train[:maxN]
        joints_test = joints_test[:(maxN/10)]

    assert I_train.shape[0] == joints_train.shape[0]
    assert I_test.shape[0] == joints_test.shape[0]

    return (I_train, I_test, joints_train, joints_test)

def perPixelLabels(I, joints, nJoints):
    '''
    scale = 30
    I_noBg = bgSub(I, joints, scale=scale)
    nJoints = joints.shape[1]
    joints_scale = joints.copy()
    joints_scale[:, :, 2] *= scale
    N = I.shape[0]

    for i in range(1):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(joints_scale[i], np.arange(0, nJoints))
        X = np.indices(I[i].shape).reshape((2, I[i].shape[0]*I[i].shape[1]))
        X = np.vstack((X, I[i].reshape(I[i].shape[0]*I[i].shape[1])*scale))
        X = X[:, X[2] != 0].T
        labels = knn.predict(X)

        img = cv2.equalizeHist(I_noBg[i].astype(np.uint8))
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #print(img)
        #for j in range(nJoints):
        #    coors = X[labels == j][:, :2].astype(int)
        #    if coors.shape[0] != 0:
        #        img[coors] = palette[j]
    '''
    cv2.imshow('image', I[0].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualizeImgs(I, joints):
    N = I.shape[0]

    for i in range(N):
        img = I[i]
        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for joint in joints[i]:
            cv2.circle(img, tuple(joint[:2].astype(np.uint16)), 2, (0,0,255), -1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawPts(img, pts):
#     img = cv2.equalizeHist(img.astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if pts.ndim == 1:
        cv2.circle(img, tuple(pts[:2].astype(np.uint16)), 4, (255,0,0), -1)
    else:
        nPts = pts.shape[0]
        for i, pt in enumerate(pts):
            color = (255*(nPts-i)/nPts, 0, 255*i/nPts)
            cv2.circle(img, tuple(pt[:2].astype(np.uint16)), 1, color, -1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawPred(img, joints, paths, center, filename, nJoints, jointName):
    H = img.shape[0]
    W = img.shape[1]

    '''
    bg = np.load('/mnt0/data/ITOP/out/00_depth_top.npy')[312]
    bg /= 1000.0
    img[img == 0] = bg[img == 0]
    '''
    img = (img-np.amin(img))*255.0/(np.amax(img)-np.amin(img))
    img = img.astype(np.uint8)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    img = np.hstack((img, np.zeros((H, 100, 3)))).astype(np.uint8)

    if paths is not None:
        paths_copy = paths.copy()
        for i, path in enumerate(paths_copy):
            nPts = path.shape[0]
            for j, pt in enumerate(path):
                color = tuple(c*(2*j+nPts)/(3*nPts) for c in palette[i])
                cv2.circle(img, tuple(pt[:2].astype(np.uint16)), 1, color, -1)

    if joints is not None:
        joints_copy = joints.copy()
        for i, joint in enumerate(joints_copy):
            cv2.circle(img, tuple(joint[:2].astype(np.uint16)), 4, palette[i], -1)

        for i, joint in enumerate(joints):
            cv2.rectangle(img, (W, int(H*i/nJoints)), (W+100, int(H*(i+1)/nJoints-1)), palette[i], -1)
            cv2.putText(img, jointName[i], (W, int(H*(i+1)/nJoints-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

    cv2.rectangle(img, tuple([int(center[0]-2), int(center[1]-2)]),
                  tuple([int(center[0]+2), int(center[1]+2)]),
                  palette[nJoints], -1)

    cv2.imwrite(filename, img)

def checkUnitVectors(unitVectors):
    s1 = np.sum(unitVectors.astype(np.float32)**2)
    s2 = unitVectors.shape[0]
    logger.debug('error: %0.3f', (abs(s1-s2)/s2))

def pixel2world(pixel, C):
    world = np.empty(pixel.shape)
    world[:, 2] = pixel[:, 2]
    world[:, 0] = (pixel[:, 0] - W/2.0) * C * pixel[:, 2]
    world[:, 1] = -(pixel[:, 1] - H/2.0) * C * pixel[:, 2]
    return world

def world2pixel(world, C):
    pixel = np.empty(world.shape)
    pixel[:, 2] = world[:, 2]
    pixel[:, 0] = (world[:, 0] / world[:, 2] / C + W / 2.0).astype(int)
    pixel[:, 1] = (-world[:, 1] / world[:, 2] / C + H / 2.0).astype(int)
    return pixel

#testing 1
#I, joints = getImgsAndJoints('/Users/alan/Documents/research/EVAL/*/joints_depthcoor/*', 200)
#print(I.shape, joints.shape)
#perPixelLabels(I, joints)
#mask = bgSub(I, joints)
#visualizeImgs(mask, joints)

#testing 2
'''
I = np.load('/mnt0/data/ITOP/out/00_depth_side.npy')
joints = np.load('/mnt0/data/ITOP/out/00_joints_side.npy')
labels = np.load('/mnt0/data/ITOP/out/00_predicts_side.npy')
print(I.shape, joints.shape, labels.shape)
visualizeImgs(labels+1, joints)
'''

#testing 3
'''
I = np.load('/Users/alan/Documents/research/healthcare/src/poseEstimation/RTW/ITOP/00_depth_side.npy')
joints = np.load('/Users/alan/Documents/research/healthcare/src/poseEstimation/RTW/ITOP/00_joints_side.npy')
labels = np.load('/Users/alan/Documents/research/healthcare/src/poseEstimation/RTW/ITOP/00_predicts_side.npy')
labels[labels >= 0] = 1
labels[labels < 0] = 0

I *= labels
print(np.mean(I[np.nonzero(I)]))
print(np.mean(I))
#visualizeImgs(I, joints)
'''

def main():
    idx = 103
    offset = 0

    view = 'top'
    test = 'top'

    qm, joint, img = None, None, None
    if os.path.isfile('visualize/qm'+str(idx)+'.npy'):
        qm = np.load('visualize/qm'+str(idx)+'.npy')
        joint = np.load('visualize/joint'+str(idx)+'.npy')
        img = np.load('visualize/img'+str(idx)+'.npy')
    else:
        qms = np.load('models_ITOP_'+view+'/qms.npy')
        joints = np.load('models_ITOP_'+view+'/joints_pred.npy')
        #I = np.load('data_ITOP_side/I_test.npy')
        i = 0
        I = None
        while True:
            I = np.load('/mnt0/data/ITOP/out/'+str(i).zfill(2)+'_depth_'+test+'.npy')
            if I.shape[0] > idx-offset:
                break
            else:
                i += 1
                offset += I.shape[0]

        if not os.path.isdir('visualize'):
            os.makedirs('visualize')
        qm = qms[idx]
        joint = joints[idx, :, :2]
        img = I[idx-offset]
        np.save('visualize/qm'+str(idx)+'.npy', qm)
        np.save('visualize/joint'+str(idx)+'.npy', joint)
        np.save('visualize/img'+str(idx)+'.npy', img)

    img *= 255.0/np.amax(img)
    img = img.astype(np.uint8)
    #img = cv2.equalizeHist(img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

    kinemOrder = None

    if test == 'top':
        kinemOrder = kinemOrderITOPUpper
    else:
        kinemOrder = kinemOrderITOP


    frameId = -1
    print(qm.shape[1])
    indices = 2*np.arange(30)
    #indices = indices**2
    #indices = indices*(float(qm.shape[1])-1)/indices[-1]
    indices = indices.astype(int)
    if not os.path.isdir('visualize/jpg'):
        os.makedirs('visualize/jpg')
    for i in kinemOrder:
        #disp = img.copy()

        for j in indices:
            pt = qm[i, j, :2].astype(int)
            cv2.circle(img, tuple(pt), 2, palette[i], -1)
            frameId += 1
            print('frameId: %d' % frameId)
            cv2.imwrite('visualize/jpg/frame_'+str(frameId).zfill(3)+'.jpg', img)#[50:, 70:230])
        #print(joint.shape)
        cv2.circle(img, tuple(joint[i].astype(int)), 2, palette[i], -1)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)


if __name__ == '__main__':
    main()
