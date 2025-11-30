import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

save_dir = './code_res_imgs/c3_featureMatching.py'

mouseDrawing = False
startX, startY, endX, endY = -1, -1, -1, -1
minMatchCount = 1
viewName = 'input image'

sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()
brisk = cv2.BRISK_create()
orb = cv2.ORB_create()

def BruteForce(img1, img2):
    methods = [(sift, cv2.NORM_L2, 'bf_sift'),
            #    (surf, cv2.NORM_L2, 'bf_surf'),
               (brisk, cv2.NORM_L2, 'bf_brisk'),
               (orb, cv2.NORM_HAMMING, 'bf_orb')]
    flag = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    for (method, dist, name) in methods:
        keyP1, des1 = method.detectAndCompute(img1, None)
        keyP2, des2 = method.detectAndCompute(img2, None)
        bf = cv2.BFMatcher_create(dist, True)
        mat = bf.match(des1, des2)
        mat = sorted(mat, key=lambda x: x.distance)
        res = cv2.drawMatches(img1, keyP1, img2, keyP2, mat, None, flags=flag)
        bfKnn = cv2.BFMatcher()
        matKnn = bfKnn.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matKnn:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        if len(good) >= minMatchCount:
            img1KP = np.float32([keyP1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            img2KP = np.float32([keyP2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _m, mask = cv2.findHomography(img1KP, img2KP, cv2.RANSAC, 5.0)
            ransacMask = mask.ravel().tolist()
            drawParams = dict(matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              matchesMask=ransacMask,
                              flags=flag)
            resKnn = cv2.drawMatchesKnn(img1, keyP1, img2, keyP2, [good], None, flags=flag)
            resKnn2 = cv2.drawMatches(img1, keyP1, img2, keyP2, good, None, **drawParams)

def flann(img1, img2):
    flannIndexKdtree = 1
    indexParamsKdtree = dict(algorithm=flannIndexKdtree, trees=5)
    searchParms = dict(checks=50)
    flannKdtree = cv2.FlannBasedMatcher(indexParamsKdtree, searchParms)
    flannIndexLsh = 6
    indexParamsLsh = dict(algorithm=flannIndexLsh,
                          table_number=6,
                          key_size=12,
                          multi_probe_level=1)
    flannLsh = cv2.FlannBasedMatcher(indexParamsLsh, searchParms)
    methods = [(sift, flannKdtree, 'flann_sift'),
            #    (surf, flannKdtree, 'flann_surf'),
               (brisk, flannLsh, 'flann_brisk'),
               (orb, flannLsh, 'flann_orb')]
    flag1 = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    flag2 = cv2.DrawMatchesFlags_DEFAULT

    for (method, matcher, name) in methods:
        keyP1, des1 = method.detectAndCompute(img1, None)
        keyP2, des2 = method.detectAndCompute(img2, None)
        mat = matcher.match(des1, des2)
        mat = sorted(mat, key = lambda x: x.distance)
        res = cv2.drawMatches(img1, keyP1, img2, keyP2, mat, None, flags = flag1)
        cv2.imshow(name, res)
        cv2.imwrite(save_dir + '/' + name + '.jpg', res)
        matKnn = matcher.knnMatch(des1, des2, k=2)
        print(name, len(matKnn))
        if len(matKnn) >= minMatchCount:
            matKnnMask = [[0, 0] for i in range(len(matKnn))]
            for i, (m, n) in enumerate(matKnn):
                if m.distance < 0.75 * n.distance:
                    matKnnMask[i] = [1, 0]
            drawParams = dict(matchColor = (0, 255, 0),
                              singlePointColor = (0, 0, 255),
                              matchesMask = matKnnMask,
                              flags = flag2)
            resKnn = cv2.drawMatchesKnn(img1, keyP1, img2, keyP2, matKnn, None, **drawParams)

def mouse_callback(event, x, y, flags, param):
    global startX, startY, mouseDrawing
    imgROI = img1.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseDrawing = True
        startX, startY = x, y
        cv2.circle(imgROI, (x, y), 2, (255, 255, 255), -1)
        cv2.imshow(viewName, imgROI)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouseDrawing:
            cv2.rectangle(imgROI, (startX, startY), (x, y), (255, 255, 255), 2)
            cv2.imshow(viewName, imgROI)
    elif event == cv2.EVENT_LBUTTONUP:
        mouseDrawing = False
        endX = x
        endY = y
        if startX > endX:
            tempX = startX
            startX = endX
            endX = tempX
        if startY > endY:
            tempY = startY
            startY = endY
            endY = tempY
        cv2.rectangle(imgROI, (startX, startY), (endX, endY), (255, 255, 255), 2)
        cv2.imshow(viewName, imgROI)
        roi = cv2.cvtColor(img1[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)
        BruteForce(roi, cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        flann(roi, cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))

img1 = cv2.imread('./img6.jpg', cv2.IMREAD_UNCHANGED)

cv2.imshow(viewName, img1)
cv2.setMouseCallback(viewName, mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()