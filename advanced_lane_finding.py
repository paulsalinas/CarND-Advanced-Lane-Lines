##
# for each correspondinog object points, obtain the corresponding image points for the corners
# add some visualization for each channel
##
#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from utils import get_calibration_points, undistorter_from_pickle
%matplotlib qt

objpoints, imgpoints = get_calibration_points()
print(objpoints, imgpoints)

##
# Apply the objpoints and imgpoints to get the calibration results
# Use the calibration results to undistort a test image. Visualize the result
##
#%%
import pickle
%matplotlib inline

# Test undistortion on an image
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None)

# Save the camera calibration result for later use (we won't worry about
# rvecs / tvecs)
pickle_path = "./wide_dist_pickle.p"
dist_pickle = {}
dist_pickle["dist"] = dist
dist_pickle["mtx"] = mtx
pickle.dump(dist_pickle, open(pickle_path, "wb"))

# Visualize undistortion

undistort_img = undistorter_from_pickle(pickle_path)
dst = undistort_img(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
f.savefig('calibration.png')


##
# define the pipeline and visualize the end result.
# steps in the pipeline:
# 1) undistort
# 2) Apply a sobel threshold filter
# 3) Apply a collor channel filter
##
# Edit this function to create your own pipeline.

#%%

from utils import pipeline

transformed = pipeline(
    undistort_img(cv2.imread('./test_images/straight_lines1.jpg')),
    sx_thresh=(20, 255),
    s_thresh=(170, 255)
)

# Plot the result
f, (ax2) = plt.subplots(1, figsize=(24, 9))
f.tight_layout()
ax2.imshow(transformed, cmap='gray')
f.savefig('pipeline.png')

##
# visualize the region of interest and the warped result
##
#%%

from utils import get_warper_from_image_shape, get_distortion_vertices_from_image_shape

image = undistort_img(cv2.imread('./test_images/straight_lines1.jpg'))

warper = get_warper_from_image_shape(image.shape)
vertices = get_distortion_vertices_from_image_shape(image.shape)

cv2.polylines(image, vertices, True, [255, 0, 255], 4)

warped, Minv = warper(image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

f.tight_layout()
ax2.imshow(warped)
ax1.imshow(image)
f.savefig('warper.png')

##
# combine the pipeline and then the warping of the image
# graph of the outcome
##
#%%
image = pipeline(undistort_img(cv2.imread(
    './test_images/straight_lines1.jpg')))
warped, _ = warper(image)
f, (ax1) = plt.subplots(1, figsize=(24, 9))
ax1.imshow(warped, cmap="gray")
f.savefig('pipe_warp.png')

##
# show a histogram of a cross section of the image
##
#%%
histogram = np.sum(warped[360:, :], axis=0)
f, (ax1) = plt.subplots(1, figsize=(24, 9))
ax1.plot(histogram)
f.savefig('histogram.png')

##
# sliding window search
##
#%%
from utils import sliding_windows, get_drawable_lanes, visualize_sliding_window

out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit = sliding_windows(
    warped)

left_fitx, right_fitx, ploty = get_drawable_lanes(
    left_fit,
    right_fit,
    warped.shape
)

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

f, (ax1) = plt.subplots(1, figsize=(24, 9))

ax1.imshow(out_img)
ax1.plot(left_fitx, ploty, color='yellow')
ax1.plot(right_fitx, ploty, color='yellow')
f.savefig('sliding_window.png')
# ax1.xlim(0, 1280)
# ax1.ylim(720, 0)


#%%
from utils import sliding_windows_lane_pixels
print(sliding_windows_lane_pixels(warped))

#%%

from utils import next_frame

left_fitx, right_fitx, left_fit, right_fit = next_frame(
    left_fit,
    right_fit,
    warped
)

#%%
from utils import visualize_lines_on_image

result = visualize_lines_on_image(ploty, left_fitx, right_fitx, warped)
plt.imshow(result)


#%%
from utils import visualize_next_step

result = visualize_next_step(
    left_fitx,
    right_fitx,
    left_lane_inds,
    right_lane_inds,
    warped
)

plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

#%%

from utils import get_curvature


print(get_curvature(left_fit))
print(get_curvature(right_fit))
