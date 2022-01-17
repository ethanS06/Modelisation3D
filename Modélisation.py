# -*- coding: utf-8 -*-
"""
Created on 2 Dec 2020

@author: chatoux
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy import ndimage
import glob

def CameraCalibration():
    # Preparation des critères d'arret de la fonction cornerSubPlix
    # Iterations = 30 et Accuracy = 0.001
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    ############ to adapt ##########################
    #Array de zeros 24 lignes et 3 colonnes
    objp = np.zeros((5*8, 3), np.float32)
    #np.mgrid 4 lignes et 6 colonnes transpose puis reshape en 2 colonnes et 1 ligne puis on les mets dans objp
    objp[:, :2] = np.mgrid[0:5, 0:8].T.reshape(-1, 2)
    # 
    objp[:, :2] *= 35 # taille de la case 40 mm
    #################################################
    # 
    objpoints = []  # 
    imgpoints = []  # 
    ############ to adapt ##########################
    images = glob.glob('Images/chess2/Nikon/*.jpeg')   # 'Images/chess/P30/*.jpg'
    #################################################
    for fname in images:
        # img = cv.imread(fname)
        #On lit l'image avec img read puis pyrDown fait un focus sur l'echiquier
        img = cv.pyrDown(cv.imread(fname))
        #Definir la couleur sur d'ont on fait l'analyse ici gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # La fonction qui trouve les corners de echiquier
        ############ to adapt ##########################
        ret, corners = cv.findChessboardCorners(gray, (5, 8), None)
        #################################################
        print(ret)
        # Si des coins son trouvé, on les ajoute à imgpoints et on ajoute à objpoints les coordonnées que nous avont specifier, on applique cornerSubPix qui
        # trouve une position plus exacte des coins, à imgpoints
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # 
            ############ to adapt ##########################
            #Dessine les corners sur l'echiquier de la fonction cornerSubPix
            cv.drawChessboardCorners(img, (5, 8), corners2, ret)
            #################################################
            cv.namedWindow('img', 0 )
            cv.imshow('img', img)
            cv.waitKey(500)
    cv.destroyAllWindows()

    # On va calibrer notre camera à l'aide de la fonction calibratecamera
    # Elle prend en parametre les coordonnees dont nous avont specifier et des corners trouves sur l'echiquier
    # mtx: La matrice de la camera qui defini la focale et le centre de la camera
    # dist: Les coefficient de distortion
    # rvecs: les vecteurs de rotations
    # tvecs: Les vecteurs de translation
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('camraMatrix\n',mtx)
    print('dist\n',dist)

    ############ to adapt ##########################
    #On choisi une image pour calibrer
    img = cv.pyrDown(cv.imread('Images/chess2/Nikon/test.jpeg')) #'Images/chess/P30/IMG_20201206_093855.jpg'
    #################################################
    # On prend la longeur et largeur de l'image en pixels
    h, w = img.shape[:2]
    #Cette fonction nous donne la nouvelle calibration de la camera puis l'affiche pour faire une comparaison avec l'originale
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print('newcameramtx\n',newcameramtx)

    # touver une fonction de mapping de l'ancienne image, et utiliser cette
    # fonction pour enlever la distortion
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # delimiter la region d'intéret et faire un crop
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.namedWindow('img', 0)
    cv.imshow('img', dst)
    cv.waitKey(0)
    ############ to adapt ##########################
    cv.imwrite('Images/calibresultM.png', dst)
    #################################################
    #On calcule l'erreur moyenne
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    return newcameramtx

def DepthMapfromStereoImages():
    ############ to adapt ##########################
    imgL = cv.pyrDown(cv.imread('Images/Jeux/left.jpg'))
    imgR = cv.pyrDown(cv.imread('Images/Jeux/right.jpg'))
    #################################################
    # crée une instance de la classe StereoSGBM qui fait correspondre des block
    # de code et non pas des pixel
    window_size = 1
    num_img_channels = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = 0,#min_disp,
    numDisparities = 16,#num_disp,
    blockSize = 3,#16,
    P1 = 8 * num_img_channels * window_size ** 2,
    P2 = 32 * num_img_channels * window_size ** 2,
    disp12MaxDiff = 16,#16,
    uniquenessRatio = 8,#10,
    speckleWindowSize = 100,#100,
    speckleRange = 32)
    # calcule de la disparity
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # Affiger l'image avec profondeur
    plt.figure('3D')
    plt.imshow((disparity-min_disp)/num_disp,'gray')
    plt.colorbar()
    plt.show()


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color, 2)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def StereoCalibrate(Cameramtx):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/Jeux/left.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/Jeux/right.jpg', 0))
    #################################################
    # opencv 4.5
    sift = cv.SIFT_create()
    # opencv 3.4
    # sift = cv.xfeatures2d.SURF_create()
    # Calculer les points d'interet les coins avec sift
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # Preparer les parametre de l'agorithme FLANN qui va faire correspondre les
    # point en s'aidant des descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # dessiner les 2 images en reliant les points qui se correspondent
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good, None, flags=2)
    plt.imshow(img3)
    plt.show()

    # Trouver la matrice essentielle à partir des appariements des points des deux images
    E, maskE = cv.findEssentialMat(pts1, pts2, Cameramtx, method=cv.FM_LMEDS)
    print('E\n',E)
    # On trouve la matrice de rotation et la matrice de translation à partir de
    # la matrice essentielle calculée
    retval, R, t, maskP = cv.recoverPose(E, pts1, pts2, Cameramtx, maskE)
    print('R\n', R)
    print('t\n', t)

    # Trouver la matrice fondamentale F (et puis FT) à partir des appariements
    # des points des deux images
    F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    print('F\n', F)

    # Calcul de la matrice fondamentale à partir de la matrice essentielle
    # Matrice Fondamentalle = K`[t]xK`xRxK-1
    FT = F

    return pts1, pts2, F, maskF, FT, maskE

def EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE):
    ############ to adapt ##########################
    img1 = cv.pyrDown(cv.imread('Images/Jeux/left.jpg', 0))
    img2 = cv.pyrDown(cv.imread('Images/Jeux/right.jpg', 0))
    #################################################
    r,c = img1.shape

    # Trouver les points d'interet en utilisant F
    pts1F = pts1[maskF.ravel() == 1]
    pts2F = pts2[maskF.ravel() == 1]

    # trouver les droite épipolaire dans l'image de droite en utilisant F
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1F,pts2F)
    # trouver les droite épipolaire dans l'image de gauche en utilisant F
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.figure('Fright')
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img6)
    plt.figure('Fleft')
    plt.subplot(121),plt.imshow(img4)
    plt.subplot(122),plt.imshow(img3)

    # Trouver les points d'interet en utilisant E
    pts1 = pts1[maskE.ravel() == 1]
    pts2 = pts2[maskE.ravel() == 1]
    #  trouver les droite épipolaire dans l'image de droite en utilisant E
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,FT)
    lines1 = lines1.reshape(-1,3)
    img5T,img6T = drawlines(img1,img2,lines1,pts1,pts2)
    plt.figure('FTright')
    plt.subplot(121),plt.imshow(img5T)
    plt.subplot(122),plt.imshow(img6T)
    # trouver les droite épipolaire dans l'image de gauche en utilisant E
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,FT)
    lines2 = lines2.reshape(-1,3)
    img3T,img4T = drawlines(img2,img1,lines2,pts2,pts1)
    plt.figure('FTleft')
    plt.subplot(121),plt.imshow(img4T)
    plt.subplot(122),plt.imshow(img3T)
    plt.show()

    # calculer les homographies qui permettent pour que les droites epipolaire soit
    # dans des lignes correspondantes
    retval, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (c,r))
    print(H1)
    print(H2)
    # Faire une transformation de perspective à l'aide des matrice de homographie
    im_dst1 = cv.warpPerspective(img1, H1, (c,r))
    im_dst2 = cv.warpPerspective(img2, H2, (c,r))
    cv.namedWindow('left', 0)
    cv.imshow('left', im_dst1)
    cv.namedWindow('right', 0)
    cv.imshow('right', im_dst2)
    cv.waitKey(0)

if __name__ == "__main__":
    #cameraMatrix = CameraCalibration()

    #cameraMatrix = np.array([[1, 0, 0],
    #                      [0, 1, 0],
    #                      [0, 0, 1]])
    # dist = [[0, 0, 0, 0, 0]]

    #pts1, pts2, F, maskF, FT, maskE = StereoCalibrate(cameraMatrix)

    #EpipolarGeometry(pts1, pts2, F, maskF, FT, maskE)

    DepthMapfromStereoImages()

