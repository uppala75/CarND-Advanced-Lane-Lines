import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points
nx = 9# number of inside corners in x
ny = 6# number of inside corners in y
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# Make a list of calibration images
images = glob.glob('./calibration*.jpg')

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
		
		# Draw and display the corners
		cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		write_name = 'corners_found'+str(idx)+'.jpg'
		cv2.imwrite(write_name,img)

#load image for reference
img = cv2.imread('./calibration1.jpg')
img_size = (img.shape[1],img.shape[0])

# Perform camera calibration with the given object and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration results for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_pickle.p", "wb"))
