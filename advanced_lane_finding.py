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

##
# visualize the region of interest and the warped result
##
#%%

from utils import get_warper_from_image_shape, get_distortion_vertices_from_image_shape

image = undistort_img(cv2.imread('./test_images/straight_lines1.jpg'))

warper = get_warper_from_image_shape(image.shape)
vertices = get_distortion_vertices_from_image_shape(image.shape)

cv2.polylines(image, vertices, True, [255, 0, 255], 4)

warped = warper(image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

f.tight_layout()
ax2.imshow(warped)
ax1.imshow(image)

##
# combine the pipeline and then the warping of the image
# graph of the outcome
##
#%%
image = pipeline(undistort_img(cv2.imread(
    './test_images/straight_lines1.jpg')))
warped = warper(image)
f, (ax1) = plt.subplots(1, figsize=(24, 9))
ax1.imshow(warped, cmap="gray")

##
# show a histogram of a cross section of the image
##
#%%
histogram = np.sum(warped[360:, :], axis=0)
f, (ax1) = plt.subplots(1, figsize=(24, 9))
ax1.plot(histogram)

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

visualize_sliding_window(
    out_img,
    left_lane_inds,
    right_lane_inds,
    nonzerox,
    nonzeroy,
    left_fitx,
    right_fitx,
    ploty
)

print(left_lane_inds)
print(nonzerox)
print(left_fit)

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


def visualize_next_step(left_fitx, right_fitx, left_lane_inds, right_lane_inds, binary_warped):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array(
        [np.transpose(
            np.vstack([left_fitx - margin, ploty])
        )]
    )

    left_line_window2 = np.array(
        [np.flipud(
            np.transpose(
                np.vstack([left_fitx + margin, ploty])
            )
        )]
    )

    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx - margin, ploty]))]
    )

    right_line_window2 = np.array(
        [np.flipud(
            np.transpose(np.vstack([right_fitx + margin, ploty]))
        )]
    )

    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result


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


# def fake_lines(leftx, rightx):
#     ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
#     # quadratic_coeff = 3e-4  # arbitrary quadratic coefficient
#     # # # For each y position generate random x position within +/-50 pix
#     # # # of the line base position in each case (x=200 for left, and x=900 for right)
#     # leftx = np.array([200 + (y**2) * quadratic_coeff + np.random.randint(-50, high=51)
#     #                   for y in ploty])
#     # rightx = np.array([900 + (y**2) * quadratic_coeff + np.random.randint(-50, high=51)
#     #                    for y in ploty])

#     # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
#     # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

#     # Fit a second order polynomial to pixel positions in each fake lane line
#     left_fit = np.polyfit(ploty, leftx, 2)
#     left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
#     right_fit = np.polyfit(ploty, rightx, 2)
#     right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

#     # Plot up the fake data
#     mark_size = 3
#     plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
#     plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
#     plt.xlim(0, 1280)
#     plt.ylim(0, 720)
#     plt.plot(left_fitx, ploty, color='green', linewidth=3)
#     plt.plot(right_fitx, ploty, color='green', linewidth=3)
#     plt.gca().invert_yaxis()  # to visualize as we do the images

#     y_eval = np.max(ploty)
#     # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#     # print(left_curverad, right_curverad# Define conversions in x and y from
#     # pixels space to meters

#     ym_per_pix = 30 / 720  # meters per pixel in y dimension
#     xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

#     # Fit new polynomials to x,y in world space
#     left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
#     right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
#     # Calculate the new radii of curvature
#     left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
#                            left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
#     right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
#                             right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
#     # Now our radius of curvature is in meters
#     print(left_curverad, 'm', right_curverad, 'm')


# fake_lines(leftx, rightx)
