from scipy.spatial import Delaunay
import numpy as np
import dlib
import cv2
import tao_asari


def rect_to_bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def shape_to_np(feature_points, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (feature_points.part(i).x, feature_points.part(i).y)

    return coords

class ForeheadCoordinateStore:
    def __init__(self,feature_points,image):
        print "Select 10 points on forehead (left to right)"
        cv2.namedWindow('Select_Forehead_Points')
        cv2.setMouseCallback('Select_Forehead_Points', self.select_point)
        self.points = np.zeros((78, 2), dtype="int")
        self.count = 0
        self.inputFeature = len(feature_points)
        self.output_marked_image = np.copy(image)
        for i in range(len(feature_points)):
            self.points[i] = feature_points[i]
        for (x, y) in feature_points:
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
        while self.count != 10:
            cv2.imshow("Select_Forehead_Points", self.output_marked_image)
            cv2.waitKey(20)
        cv2.imshow("Select_Forehead_Points", self.output_marked_image)
        cv2.waitKey(20)
        cv2.destroyWindow("Select_Forehead_Points")

    def getOutputMarkedImage(self):
        return self.output_marked_image

    def select_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print "Points selected " + str(self.count + 1)
            cv2.circle(self.output_marked_image, (x, y), 2, (0, 0, 255), -1)
            self.points[self.inputFeature + self.count] = np.array([x, y], dtype="int")
            self.count = self.count + 1
        if self.count == 10:
            print "Points successfully selected"
