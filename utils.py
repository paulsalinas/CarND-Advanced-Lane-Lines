import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def get_calibration_points(visualize=False):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if visualize:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(20)

    if visualize:
        cv2.destroyAllWindows()

    return objpoints, imgpoints


def undistorter_from_pickle(pickle_path):
    dist = pickle.load(open(pickle_path, "rb"))["dist"]
    mtx = pickle.load(open(pickle_path, "rb"))["mtx"]
    return lambda img: cv2.undistort(img, mtx, dist, None, mtx)


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    apply a sobel threshold in the x direction. 
    apply s channel threshold  
    """
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary


def get_distortion_vertices_from_image_shape(image_shape):
    """
    get the region of interest for distortion usually for visualization
    """
    # (h, w) = (image_shape[0], image_shape[1])

    # vertices = np.array([
    #     [
    #         (w // 2 - 76, h * .625),
    #         (w // 2 + 76, h * .625),
    #         (w + 100, h),
    #         (-100, h)
    #     ]
    # ], dtype=np.int32)

    img_size = image_shape

    ht_window = np.uint(img_size[0] / 1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1] / 2)

    ctl_window = c_window - .2 * np.uint(img_size[1] / 2)
    ctr_window = c_window + .2 * np.uint(img_size[1] / 2)
    cbl_window = c_window - 1 * np.uint(img_size[1] / 2)
    cbr_window = c_window + 1 * np.uint(img_size[1] / 2)

    vertices = np.array([
        [
            (cbl_window, hb_window),
            (cbr_window, hb_window),
            (ctr_window, ht_window),
            (ctl_window, ht_window)
        ]
    ], dtype=np.int32)

    return vertices


def get_distortion_shape_from_image_shape(image_shape):
    # (h, w) = (image_shape[0], image_shape[1])

    img_size = image_shape

    ht_window = np.uint(img_size[0] / 1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1] / 2)

    ctl_window = c_window - .2 * np.uint(img_size[1] / 2)
    ctr_window = c_window + .2 * np.uint(img_size[1] / 2)
    cbl_window = c_window - 1 * np.uint(img_size[1] / 2)
    cbr_window = c_window + 1 * np.uint(img_size[1] / 2)

    src = np.float32([
        [cbl_window, hb_window],
        [cbr_window, hb_window],
        [ctr_window, ht_window],
        [ctl_window, ht_window]
    ])

    dst = np.float32([
        [0, img_size[0]],
        [img_size[1], img_size[0]],
        [img_size[1], 0],
        [0, 0]
    ])

    # src = np.float32([
    #     [w // 2 - 76, h * .625],
    #     [w // 2 + 76, h * .625],
    #     [-100, h],
    #     [w + 100, h]
    # ])

    # # Define corresponding destination points
    # dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

    return src, dst


def get_warper_from_image_shape(image_shape):
    """
    gets a warper function from the given image shape
    """
    src, dst = get_distortion_shape_from_image_shape(image_shape)
    return lambda img: warper(img, src, dst)


def warper(img, src, dst):
    """
    warp the given image given the source dimensions and destination dimensionså
    """

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped


def get_drawable_lanes(left_fit, right_fit, binary_warped_shape):
    ploty = np.linspace(0, binary_warped_shape[0] - 1, binary_warped_shape[0])

    # drawable left_fit and right_fit
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def visualize_sliding_window(out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, left_fitx, right_fitx, ploty):
    # plot this seperately
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # f, (ax1) = plt.subplots(1, figsize=(24, 9))

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


# def get_lane_polyfit(nonzerox, nonzeroy, )


def sliding_windows(binary_warped):
    """
    takes a warped image and return of tuple of (left lane indices, right lane indices)  
    """

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(
        binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]

        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # left_fitx, right_fitx, ploty = get_drawable_lanes(
    #     left_fit,
    #     right_fit,
    #     binary_warped.shape
    # )

    return out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit


def next_frame(left_fit, right_fit, binary_warped):

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100

    left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox < (left_fit[0] * (nonzeroy**2) +
                     left_fit[1] * nonzeroy + left_fit[2] + margin))
    )

    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox < (right_fit[0] * (nonzeroy**2) +
                     right_fit[1] * nonzeroy + right_fit[2] + margin))
    )

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, left_fit, right_fit


def draw_lines_on_warped(warped, left_fitx, right_fitx):
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

    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def visualize_lines_on_image(ploty, left_fitx, right_fitx, binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    margin = 100

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
