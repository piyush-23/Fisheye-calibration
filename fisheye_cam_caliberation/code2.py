import cv2
import numpy as np
import os
import glob
from configparser import ConfigParser


# calibrating the master image

# Defining the dimensions of checkerboard
CHECKERBOARD_M = (10,15)
criteria_m = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points checkerboard image
objpoints_m = []
# Creating vector to store vectors of 2D points checkerboard image
imgpoints_m = [] 


# Defining the world coordinates for 3D points
objp_m = np.zeros((1, CHECKERBOARD_M[0]*CHECKERBOARD_M[1], 3), np.float32)
objp_m[0,:,:2] = np.mgrid[0:CHECKERBOARD_M[0], 0:CHECKERBOARD_M[1]].T.reshape(-1, 2)
prev_img_shape_m = None


#reading the image
img_m = cv2.imread('./images/master.jpg')
gray_m = cv2.cvtColor(img_m,cv2.COLOR_BGR2GRAY)
# Find the chess board corners
# If desired number of corners are found in the image then ret = true
ret_m, corners_m = cv2.findChessboardCorners(gray_m, CHECKERBOARD_M, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

"""
If desired number of corner are detected,
we refine the pixel coordinates and display 
them on the image of checker board
"""
if ret_m == True:
    objpoints_m.append(objp_m)
    # refining pixel coordinates for given 2d points.
    corners2_m = cv2.cornerSubPix(gray_m,corners_m,(11,11),(-1,-1),criteria_m)
        
    imgpoints_m.append(corners2_m)

    # Draw and display the corners
    img_m = cv2.drawChessboardCorners(img_m, CHECKERBOARD_M, corners2_m,ret_m)    
cv2.imshow('master_img',img_m)
cv2.waitKey(0)    

cv2.destroyAllWindows()

img_shape_m = img_m.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_m, mtx_m, dist_m, rvecs_m, tvecs_m = cv2.calibrateCamera(objpoints_m, imgpoints_m, gray_m.shape[::-1],None,None)

#calculating the calibration result for master image
print("Calibration result of master image : \n")
print(ret_m)
'''
print("Camera matrix : \n")
print(mtx_m)
print("Distortion Coefficients : \n")
print(dist_m)
print("Rotational vector : \n")
print(rvecs_m)
print("Translational vector : \n")
print(tvecs_m)
'''


#calibrating the test image and undistort it if needed

# Defining the dimensions of checkerboard
CHECKERBOARD_t = (6,9)
criteria_t = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points checkerboard image
objpoints_t = []
# Creating vector to store vectors of 2D points checkerboard image
imgpoints_t = [] 


# Defining the world coordinates for 3D points
objp_t = np.zeros((1, CHECKERBOARD_t[0]*CHECKERBOARD_t[1], 3), np.float32)
objp_t[0,:,:2] = np.mgrid[0:CHECKERBOARD_t[0], 0:CHECKERBOARD_t[1]].T.reshape(-1, 2)
prev_img_shape_t = None


#reading the image
img_t = cv2.imread('./images/test3.jpg')
gray_t = cv2.cvtColor(img_t,cv2.COLOR_BGR2GRAY)
# Find the chess board corners
# If desired number of corners are found in the image then ret = true
ret_t, corners_t = cv2.findChessboardCorners(gray_t, CHECKERBOARD_t, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

"""
If desired number of corner are detected,
we refine the pixel coordinates and display 
them on the image of checker board
"""
if ret_t == True:
    objpoints_t.append(objp_t)
    # refining pixel coordinates for given 2d points.
    corners2_t = cv2.cornerSubPix(gray_t,corners_t,(11,11),(-1,-1),criteria_t)
        
    imgpoints_t.append(corners2_t)

    # Draw and display the corners
    img_t = cv2.drawChessboardCorners(img_t, CHECKERBOARD_t, corners2_t,ret_t)    
cv2.imshow('test_img',img_t)
cv2.waitKey(0)    

cv2.destroyAllWindows()

img_shape_t = img_t.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret_t, mtx_t, dist_t, rvecs_t, tvecs_t = cv2.calibrateCamera(objpoints_t, imgpoints_t, gray_t.shape[::-1],None,None)

#printing all the results:
print("Calibration result : \n")
print(ret_t)
print("Camera matrix : \n")
print(mtx_t)
print("Distortion Coefficients : \n")
print(dist_t)
print("Rotational vector : \n")
print(rvecs_t)
print("Translational vector : \n")
print(tvecs_t)


'''
here we are comparing the values of ret_t and ret_m value of testing and master image,
if ret_t of test image(i.e. test1.jpg) is greater than the ret_m value of master image(i.e. master.jpg),
then we will undistort the image otherwise not.
'''

if ret_t > ret_m :
    # undistortion code
    # Using the derived camera parameters to undistort the image
    img_t = cv2.imread('./images/test3.jpg')
    h_t,w_t = img_t.shape[:2]
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx_t, roi_t = cv2.getOptimalNewCameraMatrix(mtx_t, dist_t, img_shape_t, 1, img_shape_t)

    dst_t = cv2.undistort(img_t, mtx_t, dist_t, None, newcameramtx_t)

    mapx_t,mapy_t = cv2.initUndistortRectifyMap(mtx_t,dist_t,None,newcameramtx_t,(w_t,h_t),5)
    dst_t = cv2.remap(img_t,mapx_t,mapy_t,cv2.INTER_LINEAR)

    cv2.imshow("Undistorted Image",dst_t)
    cv2.waitKey(0)