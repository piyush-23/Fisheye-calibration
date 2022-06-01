import cv2
import numpy as np
import os
import glob
from configparser import ConfigParser
import json
from json import JSONEncoder

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of for each checkerboard image
objpoints = []
# Creating vector to store vectors of for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


#reading the image
img = cv2.imread('./images/test3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Find the chess board corners
# If desired number of corners are found in the image then ret = true
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

"""
If desired number of corner are detected,
we refine the pixel coordinates and display 
them on the images of checker board
"""
if ret == True:
    objpoints.append(objp)
    # refining pixel coordinates for given 2d points.
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)    
cv2.imshow('img',img)
cv2.waitKey(0)    

cv2.destroyAllWindows()

img_shape = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


#printing all the results:
print("Calibration result : \n")
print(ret)
print("Camera matrix : \n")
print(mtx)
print("Distortion Coefficients : \n")
print(dist)
print("Rotational vector : \n")
print(rvecs)
print("Translational vector : \n")
print(tvecs)

#converting the (tvecs,rvecs) numpy array to simple list
tvecs = np.array(tvecs).tolist()
rvecs = np.array(rvecs).tolist()

#labelling the names
test_names = ['Translational Coordinate',
            'Rotational Coordinate'] 
tvector = {test_names[0]: tvecs}           
rvector = {test_names[1]: rvecs}
#creating the json file and adding the coordinate of rvec and tvec in it.                       
with open('data.json', 'w') as f:
    json.dump(tvector, f,separators=(',', ':'))
    f.write('\n')
    json.dump(rvector, f,separators=(',', ':'))

'''
We are storing the ret value of master image in config.ini file as RET_VALUE,
and after that we will compare that RET_value with ret of test image(i.e. test1.jpg)
if ret of test image(i.e. test1.jpg) is greater than the RET_VALUE of master image(i.e. master.jpg),
then we will undistort the image otherwise not.
'''
#reading the config.ini file
config = ConfigParser()
config.read('./config.ini')

RET_VALUE = float(config['CAMERAPARAMETER']['RET_VALUE'])

if ret > RET_VALUE :
    # undistortion code
    # Using the derived camera parameters to undistort the image
    img = cv2.imread('./images/test3.jpg')
    h,w = img.shape[:2]
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    cv2.imshow("Undistorted Image",dst)
    cv2.waitKey(0)
