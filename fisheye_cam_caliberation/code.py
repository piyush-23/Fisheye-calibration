import cv2
import numpy as np
import os
import glob
from configparser import ConfigParser

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points checkerboard image
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
    # print("coordinates",imgpoints)
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

"""
calibrateCamera returns the root mean square (RMS) re-projection error(ret value),
usually it should be between 0.1 and 1.0 pixels in a good calibration.
and above 1.0 is bad calibration.
link for this : https://stackoverflow.com/questions/29628445/meaning-of-the-retval-return-value-in-cv2-calibratecamera
"""

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

'''
We will undistort the image when it's ret value is greater then 1,
otherwise no need to undistort the image.
'''

if ret > 1:
    # undistortion code
    # Using the derived camera parameters to undistort the image
    # We can use any image from range 0-40 to undistort
    img = cv2.imread('./images/test3.jpg')
    h,w = img.shape[:2]
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

    cv2.imshow("Undistorted Image",dst)
    cv2.waitKey(0)


'''
# undistortion code
# Using the derived camera parameters to undistort the image
# We can use any image from range 0-40 to undistort
img = cv2.imread('./images/test3.jpg')
h,w = img.shape[:2]
# Refining the camera matrix using parameters obtained by calibration
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

cv2.imshow("Undistorted Image",dst)
cv2.waitKey(0)
'''