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
    (h, w) = (image_shape[0], image_shape[1])

    vertices = np.array([
        [
            (w // 2 - 76, h * .625),
            (w // 2 + 76, h * .625),
            (w + 100, h),
            (-100, h)
        ]
    ], dtype=np.int32)

    return vertices


def get_distortion_shape_from_image_shape(image_shape):
    (h, w) = (image_shape[0], image_shape[1])

    src = np.float32([
        [w // 2 - 76, h * .625],
        [w // 2 + 76, h * .625],
        [-100, h],
        [w + 100, h]
    ])

    # Define corresponding destination points
    dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

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
