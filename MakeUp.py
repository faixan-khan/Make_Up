import cv2
import sys
import numpy as np
import FeatureDetection
import wsFilter


image_height = 500

sourceImagePath = "./SampleImages/source.jpg"
sourceImage = cv2.imread(sourceImagePath)
r = image_height * 1.0 / sourceImage.shape[0]
dim = (int(sourceImage.shape[1] * r), image_height)
sourceImage = cv2.resize(sourceImage, dim, interpolation=cv2.INTER_AREA)

output_image1, feature_points1, triangulation1,K1 = FeatureDetection.landmark_detection(sourceImage)
sourceTuple = (output_image1, feature_points1, triangulation1,K1)
Kdash = (K1 - np.min(K1))/(np.max(K1)-np.min(K1))
cv2.imshow("Face Mask",Kdash)

makeUpImagePath = "./SampleImages/makeU.jpg"
makeUpImage = cv2.imread(makeUpImagePath)
r = image_height * 1.0 / makeUpImage.shape[0]
dim = (int(makeUpImage.shape[1] * r), image_height)
makeUpImage = cv2.resize(makeUpImage, dim, interpolation=cv2.INTER_AREA)
output_image2, feature_points2, triangulation2,K2 = FeatureDetection.landmark_detection(makeUpImage)
makeUpTuple = (output_image2, feature_points2, triangulation2,K2)

sourceLAB = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2LAB)
resultant_image = np.copy(sourceLAB)

sourceLightness = sourceLAB[:,:,0]
sourceColor = sourceLAB[:,:,1:3]
sourceStructureLayer,sourceTextureLayer = wsFilter.wls_filter(sourceLightness,K1)

makeUpLAB = cv2.cvtColor(makeUpImage,cv2.COLOR_BGR2LAB)
makeUpLightness = makeUpLAB[:,:,0]
makeUpColor = makeUpLAB[:,:,1:3]
makeUpStructureLayer,makeUpTextureLayer = wsFilter.wls_filter(makeUpLightness,K2)
