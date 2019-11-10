import cv2
import sys
import numpy as np
import FeatureDetection
import wsFilter
import morphing

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


deltaInput = 0
deltaMakeUp = 1
resultant_skin_detail = morphing.morph_image_delta1_delta2(sourceTextureLayer,makeUpTextureLayer,deltaInput,deltaMakeUp,sourceTuple,makeUpTuple)

alphaFactor = 0.8
resultant_image[:,:,1:3] = morphing.morph_image_delta1_delta2(sourceColor,makeUpColor,1 - alphaFactor,alphaFactor,sourceTuple,makeUpTuple,True)

scaleFactor = 0.5
sourceStructureLayer_halved = cv2.resize(sourceStructureLayer,(int(sourceStructureLayer.shape[1]*scaleFactor),int(sourceStructureLayer.shape[0]*scaleFactor)),interpolation=cv2.INTER_AREA)
dim = (sourceStructureLayer.shape[1],sourceStructureLayer.shape[0])
sourceStructureLayer_retained_blurred = cv2.resize(sourceStructureLayer_halved,dim,interpolation=cv2.INTER_AREA)


laplacian_makeUpStructure = cv2.Laplacian(makeUpStructureLayer,cv2.CV_64F)

resultant_structure_layer = morphing.morph_image_delta1_delta2(sourceStructureLayer_retained_blurred,laplacian_makeUpStructure,1,1,sourceTuple,makeUpTuple)

sum_detail_structure = resultant_skin_detail + resultant_structure_layer
sum_detail_structure = (sum_detail_structure - np.min(sum_detail_structure))/(np.max(sum_detail_structure)-np.min(sum_detail_structure))

resultant_image[:,:,0] = (sum_detail_structure*255).astype(np.uint8)

resultant_image = cv2.cvtColor(resultant_image,cv2.COLOR_LAB2BGR)

cv2.imshow("Output",np.hstack((sourceImage,resultant_image,makeUpImage)))
key = cv2.waitKey(0)
if key & 0xFF == ord('s'):
    cv2.imwrite('out.jpg',np.hstack((sourceImage,resultant_image,makeUpImage)))
