# Fisheye-calibration

1. In code.py file there is simple calibration and undistortion of one fisheye image.

2. In code1.py file we are comparing the values of ret value of master image and test image,
means we are storing the ret value in config.ini file as RET_VALUE,
and after that we will compare the RET_value with ret of test image(i.e. test1.jpg)
if ret of test image(i.e. test1.jpg) is greater than the ret value of master image(i.e. master.jpg),
then we will undistort the image otherwise not.

3. In code2.py file we are comparing the values of ret_t and ret_m value of testing and master image,
if ret_t of test image(i.e. test1.jpg) is greater than the ret_m value of master image(i.e. master.jpg),
then we will undistort the image otherwise not.
