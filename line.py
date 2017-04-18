import numpy as np
from collections import deque
from utils import sliding_windows_lane_pixels, next_frame_search, polyfit, calc_radius


class LaneLines():
    def __init__(self):
        self.num_searches = 0
        self.last_successful_search = 0
        self.right_line = Line()
        self.left_line = Line()

    def process_warped_image(self, img):
        self.num_searches = self.num_searches + 1

        left_fit = None
        right_fit = None

        if not self.left_line.detected and not self.right_line.detected:
            # start with a sliding window search if no lines are detected
            self.sliding_window_search(img)
            self.left_line.detected = True
            self.right_line.detected = True
            return
        else:
            # do a next frame search if lines are detected based on existing
            # fits
            leftx, lefty, rightx, righty = next_frame_search(
                self.left_line.current_fit,
                self.right_line.current_fit,
                img
            )

            left_fit, right_fit = polyfit(leftx, lefty, rightx, righty)

            left_fit_curve, right_fit_curve = calc_radius(
                img, leftx, lefty, rightx, righty)

            # assess
            # do the curves make sense? are they extreme?
            def is_extreme_curve(curve): return curve < 0.1 or curve > 10

            if (is_extreme_curve(left_fit_curve) or is_extreme_curve(right_fit_curve)):
                self.left_line.detected = False
                self.right_line.detected = False
                return

            self.left_line.add_fit(left_fit)
            self.right_line.add_fit(right_fit)

            self.left_line.radius_of_curvature = left_fit_curve
            self.right_line.radius_of_curvature = right_fit_curve

    def sliding_window_search(self, img):
        leftx, lefty, rightx, righty = sliding_windows_lane_pixels(img)
        left_fit, right_fit = polyfit(leftx, lefty, rightx, righty)

        self.left_line.add_fit(left_fit)
        self.right_line.add_fit(right_fit)

        left_fit_curve, right_fit_curve = calc_radius(
            img, leftx, lefty, rightx, righty)

        avg_curve = (left_fit_curve + right_fit_curve) / 2

        self.left_line.radius_of_curvature = left_fit_curve
        self.right_line.radius_of_curvature = right_fit_curve


class Line():
    def __init__(self):

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line

        self.last_n_fits = deque(iterable=[], maxlen=10)

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def add_fit(self, fit):
        self.current_fit = fit
        self.last_n_fits.append(fit)
        self.best_fit = np.average(self.last_n_fits, axis=0)
