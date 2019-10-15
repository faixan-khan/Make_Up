import cv2
import sys
import numpy as np

image_height = 500

sourceImagePath = "./SampleImages/source.jpg"
sourceImage = cv2.imread(sourceImagePath)
r = image_height * 1.0 / sourceImage.shape[0]
dim = (int(sourceImage.shape[1] * r), image_height)
sourceImage = cv2.resize(sourceImage, dim, interpolation=cv2.INTER_AREA)

#Implement Feature FeatureDetection

makeUpImagePath = "./SampleImages/makeU.jpg"
makeUpImage = cv2.imread(makeUpImagePath)
r = image_height * 1.0 / makeUpImage.shape[0]
dim = (int(makeUpImage.shape[1] * r), image_height)
makeUpImage = cv2.resize(makeUpImage, dim, interpolation=cv2.INTER_AREA)

#Implement Feature Detection
