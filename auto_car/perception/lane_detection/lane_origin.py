import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
from perception.lane_detection import edge_detection as edge  # Handles the detection of lane lines
# import edge_detection as edge  # Handles the detection of lane lines
import matplotlib.pyplot as plt  # Used for plotting and error checking

import os  # Import the os library to handle file paths

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Implementation of the Lane class

# Make sure the video file is in the same directory as your code
filename = "orig_lane_detection_1.mp4"
file_size = (1920, 1080)  # Assumes 1920x1080 mp4
scale_ratio = 1  # Option to scale to fraction of original size.

# We want to save the output to a video file
output_filename = "orig_lane_detection_1_lanes.mp4"
output_frames_per_second = 20.0

# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []


class Lane:
    """
    Represents a lane on a road.
    """

    def __init__(self, orig_frame):
        """
              Default constructor

        :param orig_frame: Original camera image (i.e. frame)
        """
        self.orig_frame = orig_frame

        # This will hold an image with the lane lines
        self.lane_line_markings = None

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        # Four corners of the trapezoid-shaped region of interest
        # You need to find these corners manually.
        self.roi_points = np.float32(
            [
                (int(0.294375 * width), int(0.108333 * height)),  # Top-left corner
                (int(0.025313 * width), int(0.508333 * height)),  # Bottom-left corner
                (int(0.950938 * width), int(0.469417 * height)),  # Bottom-right corner
                (int(0.682187 * width), int(0.077083 * height)),  # Top-right corner
            ]
        )

        # The desired corner locations  of the region of interest
        # after we perform perspective transformation.
        # Assume image width of 600, padding == 150.
        self.padding = int(0.25 * width)  # padding from side of the image in pixels
        self.desired_roi_points = np.float32(
            [
                [self.padding, 0],  # Top-left corner
                [self.padding, self.orig_image_size[1]],  # Bottom-left corner
                [
                    self.orig_image_size[0] - self.padding,
                    self.orig_image_size[1],
                ],  # Bottom-right corner
                [self.orig_image_size[0] - self.padding, 0],  # Top-right corner
            ]
        )

        self.sign_block_points = np.float32(
            [
                (int(0.348438 * width), int(0.175833 * height)),  # Top-left corner
                (int(0.325000 * width), int(0.330833 * height)),  # Bottom-left corner
                (int(0.646875 * width), int(0.320417 * height)),  # Bottom-right corner
                (int(0.631250 * width), int(0.177917 * height)),  # Top-right corner
            ]
        )

        # Histogram that shows the white pixel peaks for lane line detection
        self.histogram = None

        # Sliding window parameters
        self.no_of_windows = 10
        self.margin = int((1 / 12) * width)  # Window width is +/- margin
        self.minpix = int((1 / 24) * width)  # Min no. of pixels to recenter window

        # Best fit polynomial lines for left line and right line of the lane
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 7.0 / 400  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 255  # meters per pixel in x dimension

        # Radii of curvature and offset
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None

    def calculate_car_position(self, print_to_terminal=False):
        """
        Calculate the position of the car relative to the center

        :param: print_to_terminal Display data to console if True
        :return: Offset from the center of the lane
        """
        # Assume the camera is centered in the image.
        # Get position of car in centimeters
        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = (
            self.left_fit[0] * height**2 + self.left_fit[1] * height + self.left_fit[2]
        )
        bottom_right = (
            self.right_fit[0] * height**2
            + self.right_fit[1] * height
            + self.right_fit[2]
        )

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (
            (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100
        )

        if print_to_terminal == True:
            print(str(center_offset) + "cm")

        self.center_offset = center_offset

        return center_offset

    def calculate_curvature(self, print_to_terminal=False):
        """
        Calculate the road curvature in meters.

        :param: print_to_terminal Display data to console if True
        :return: Radii of curvature
        """
        # Set the y-value where we want to calculate the road curvature.
        # Select the maximum y-value, which is the bottom of the frame.
        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(
            self.lefty * self.YM_PER_PIX, self.leftx * (self.XM_PER_PIX), 2
        )
        right_fit_cr = np.polyfit(
            self.righty * self.YM_PER_PIX, self.rightx * (self.XM_PER_PIX), 2
        )

        # Calculate the radii of curvature
        left_curvem = (
            (1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * left_fit_cr[0])
        right_curvem = (
            (
                1
                + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1])
                ** 2
            )
            ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal == True:
            print(left_curvem, "m", right_curvem, "m")

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    def calculate_histogram(self, frame=None, plot=True):
        """
        Calculate the image histogram to find peaks in white pixel count

        :param frame: The warped image
        :param plot: Create a plot if True
        """
        if frame is None:
            frame = self.warped_frame

        # Generate the histogram
        self.histogram = np.sum(frame[int(frame.shape[0] / 2) :, :], axis=0)

        if plot == True:

            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap="gray")
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def display_curvature_offset(self, frame=None, plot=False):
        """
        Display curvature and offset statistics on the image

        :param: plot Display the plot if True
        :return: Image with lane lines and curvature
        """
        image_copy = None
        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(
            image_copy,
            "Curve Radius: "
            + str((self.left_curvem + self.right_curvem) / 2)[:7]
            + " m",
            (int((5 / 600) * self.width), int((20 / 338) * self.height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            (float((0.5 / 600) * self.width)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_copy,
            "Center Offset: " + str(self.center_offset)[:7] + " cm",
            (int((5 / 600) * self.width), int((40 / 338) * self.height)),
            cv2.FONT_HERSHEY_SIMPLEX,
            (float((0.5 / 600) * self.width)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if plot == True:
            cv2.imshow("Image with Curvature and Offset", image_copy)

        return image_copy

    def compute_turn_command(
        self,
        wheelbase_m: float = 2.8,
        max_steer_deg: float = 30.0,
        deg_deadband: float = 1.0,
        offset_gain: float = 0.05,
        lookahead_m: float = 15.0,
    ):
        """
        根據偵測到的車道線，計算轉向建議。

        回傳: (direction, steer_deg, signed_radius_m)
          - direction: 'left' | 'right' | 'straight'
          - steer_deg: 推薦方向盤角度（度，左正右負，已限幅）
          - signed_radius_m: 有號曲率半徑（m），左彎為正，右彎為負；直線回傳 inf
        """
        # 確保有足夠資料
        if (
            self.ploty is None
            or self.leftx is None
            or self.rightx is None
            or self.lefty is None
            or self.righty is None
        ):
            return "straight", 0.0, float("inf")

        # 以公尺為單位重算兩條線（x = a y^2 + b y + c）
        left_fit_cr = np.polyfit(
            self.lefty * self.YM_PER_PIX, self.leftx * self.XM_PER_PIX, 2
        )
        right_fit_cr = np.polyfit(
            self.righty * self.YM_PER_PIX, self.rightx * self.XM_PER_PIX, 2
        )

        # 中心線係數（左右平均）
        a_c = 0.5 * (left_fit_cr[0] + right_fit_cr[0])
        b_c = 0.5 * (left_fit_cr[1] + right_fit_cr[1])

        # 在畫面底部 y 評估有號曲率 kappa
        y_eval_m = float(np.max(self.ploty)) * self.YM_PER_PIX
        denom = (1.0 + (2.0 * a_c * y_eval_m + b_c) ** 2) ** 1.5
        if denom < 1e-6:
            denom = 1e-6
        kappa = 2.0 * a_c / denom  # >0 左彎，<0 右彎（依本坐標定義）

        # 由曲率推 steering：delta = atan(L * kappa)
        steer_rad = np.arctan(wheelbase_m * kappa)

        # 加入中心偏移校正（center_offset 以 cm 計）
        offset_m = (self.center_offset or 0.0) / 100.0
        steer_rad += np.arctan2(offset_gain * offset_m, max(lookahead_m, 1e-3))

        steer_deg = float(np.degrees(steer_rad))
        steer_deg = float(np.clip(steer_deg, -max_steer_deg, max_steer_deg))

        # 方向與有號半徑
        direction = (
            "left" if steer_deg > deg_deadband else "right" if steer_deg < -deg_deadband else "straight"
        )
        signed_radius_m = float("inf") if abs(kappa) < 1e-6 else 1.0 / kappa

        return direction, steer_deg, signed_radius_m

    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """
        # margin is a sliding window parameter
        margin = self.margin

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = (
            nonzerox
            > (
                # left_fit[0] * (nonzeroy**2)
                # + left_fit[1] * nonzeroy
                # + left_fit[2]
                np.polyval(left_fit, nonzeroy)
                - margin
            )
        ) & (
            nonzerox
            < (
                # left_fit[0] * (nonzeroy**2)
                # + left_fit[1] * nonzeroy
                # + left_fit[2]
                np.polyval(left_fit, nonzeroy)
                + margin
            )
        )
        right_lane_inds = (
            nonzerox
            > (
                # right_fit[0] * (nonzeroy**2)
                # + right_fit[1] * nonzeroy
                # + right_fit[2]
                np.polyval(right_fit, nonzeroy)
                - margin
            )
        ) & (
            nonzerox
            < (
                # right_fit[0] * (nonzeroy**2)
                # + right_fit[1] * nonzeroy
                # + right_fit[2]
                np.polyval(right_fit, nonzeroy)
                + margin
            )
        )
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Get the left and right lane line pixel locations
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        global prev_leftx2
        global prev_lefty2
        global prev_rightx2
        global prev_righty2
        global prev_left_fit2
        global prev_right_fit2

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx2
            lefty = prev_lefty2
            rightx = prev_rightx2
            righty = prev_righty2

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        prev_left_fit2.append(left_fit)
        prev_right_fit2.append(right_fit)

        # Calculate the moving average
        if len(prev_left_fit2) > 10:
            prev_left_fit2.pop(0)
            prev_right_fit2.pop(0)
            left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
            right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx2 = leftx
        prev_lefty2 = lefty
        prev_rightx2 = rightx
        prev_righty2 = righty

        # Create the x and y values to plot on the image
        ploty = np.linspace(
            0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0]
        )
        # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        left_fitx = np.polyval(left_fit, ploty)
        # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        right_fitx = np.polyval(right_fit, ploty)
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot == True:

            # Generate images to draw on
            out_img = (
                np.dstack((self.warped_frame, self.warped_frame, (self.warped_frame)))
                * 255
            )
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # Create a polygon to show the search window area, and recast
            # the x and y points into a usable format for cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array(
                [np.transpose(np.vstack([left_fitx - margin, ploty]))]
            )
            left_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))]
            )
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array(
                [np.transpose(np.vstack([right_fitx - margin, ploty]))]
            )
            right_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))]
            )
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.array([left_line_pts], dtype=np.int32), (0, 255, 0))
            cv2.fillPoly(window_img, np.array([right_line_pts], dtype=np.int32), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap="gray")
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color="yellow")
            ax3.plot(right_fitx, ploty, color="yellow")
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        """
        Get the indices of the lane line pixels using the
        sliding windows technique.

        :param: plot Show plot or not
        :return: Best fit lines for the left and right lines of the current lane
        """
        # Sliding window width is +/- margin
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Set the height of the sliding windows
        window_height = int(self.warped_frame.shape[0] / self.no_of_windows)

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(
                frame_sliding_window,
                (win_xleft_low, win_y_low),
                (win_xleft_high, win_y_high),
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                frame_sliding_window,
                (win_xright_low, win_y_low),
                (win_xright_high, win_y_high),
                (255, 255, 255),
                2,
            )

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = None
        right_fit = None

        global prev_leftx
        global prev_lefty
        global prev_rightx
        global prev_righty
        global prev_left_fit
        global prev_right_fit

        # Make sure we have nonzero pixels
        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
            leftx = prev_leftx
            lefty = prev_lefty
            rightx = prev_rightx
            righty = prev_righty
            
        # x and y are None, means no lane lines detected
        if leftx is None or lefty is None or rightx is None or righty is None:
            return None, None
        

        # left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)

        # Add the latest polynomial coefficients
        # prev_left_fit.append(left_fit)
        # prev_right_fit.append(right_fit)

        # Calculate the moving average
        # if len(prev_left_fit) > 10:
        #     prev_left_fit.pop(0)
        #     prev_right_fit.pop(0)
        #     left_fit = sum(prev_left_fit) / len(prev_left_fit)
        #     right_fit = sum(prev_right_fit) / len(prev_right_fit)

        # 判斷點數門檻（可依資料量調整）
        min_pts_quadratic = 12   # 二次擬合建議下限
        min_pts_linear    = 6    # 一次擬合建議下限

        def fit_quadratic_or_linear(y, x):
            n = len(x)
            if n >= min_pts_quadratic:
                coef = np.polyfit(y, x, 2)  # [a2, a1, a0]
                return coef, 2
            elif n >= min_pts_linear:
                a1, a0 = np.polyfit(y, x, 1)  # x = a1*y + a0
                return np.array([0.0, a1, a0], dtype=np.float64), 1  # 填成二次形
            else:
                return None, 0

        left_result = fit_quadratic_or_linear(lefty, leftx)
        right_result = fit_quadratic_or_linear(righty, rightx)

        left_fit, left_deg   = left_result
        right_fit, right_deg = right_result

        # 若點數仍不足，直接使用上一幀的係數，不更新歷史，避免污染
        if left_fit is None:
            left_fit = getattr(self, "left_fit", None)
        if right_fit is None:
            right_fit = getattr(self, "right_fit", None)
        if left_fit is None or right_fit is None:
            return None, None

        # 指數移動平均（EMA）比簡單平均更抗突變且延遲更小
        alpha = getattr(self, "ema_alpha", 0.2)  # 可在 __init__ 設 self.ema_alpha

        if not hasattr(self, "left_fit_ema") or self.left_fit_ema is None:
            self.left_fit_ema = left_fit.astype(np.float64)
        else:
            self.left_fit_ema = alpha * left_fit + (1.0 - alpha) * self.left_fit_ema

        if not hasattr(self, "right_fit_ema") or self.right_fit_ema is None:
            self.right_fit_ema = right_fit.astype(np.float64)
        else:
            self.right_fit_ema = alpha * right_fit + (1.0 - alpha) * self.right_fit_ema

        left_fit = self.left_fit_ema
        right_fit = self.right_fit_ema

        self.left_fit = left_fit
        self.right_fit = right_fit

        prev_leftx = leftx
        prev_lefty = lefty
        prev_rightx = rightx
        prev_righty = righty

        if plot == True:

            # Create the x and y values to plot on the image
            ploty = np.linspace(
                0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0]
            )
            # left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            left_fitx = np.polyval(left_fit, ploty)
            # right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
            right_fitx = np.polyval(right_fit, ploty)

            # Generate an image to visualize the result
            out_img = (
                np.dstack(
                    (frame_sliding_window, frame_sliding_window, (frame_sliding_window))
                )
                * 255
            )

            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Plot the figure with the sliding windows
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap="gray")
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color="yellow")
            ax3.plot(right_fitx, ploty, color="yellow")
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit

    def get_line_markings(self, frame=None):
        """
        Isolates lane lines.

              :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary (i.e. black and white) image containing the lane lines.
        """
        if frame is None:
            frame = self.orig_frame

        # Convert the video frame from BGR (blue, green, red)
        # color space to HLS (hue, saturation, lightness).
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        ################### Isolate possible lane line edges ######################

        # Perform Sobel edge detection on the L (lightness) channel of
        # the image to detect sharp discontinuities in the pixel intensities
        # along the x and y axis of the video frame.
        # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
        # Relatively light pixels get made white. Dark pixels get made black.
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)  # Reduce noise

        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        ######################## Isolate possible lane lines ######################

        # Perform binary thresholding on the S (saturation) channel
        # of the video frame. A high saturation value means the hue color is pure.
        # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
        # and have high saturation channel values.
        # s_binary is matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the purest hue colors (e.g. >130...play with
        # this value for best results).
        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = edge.threshold(s_channel, (130, 255))

        # Perform binary thresholding on the R (red) channel of the
        # original BGR video frame.
        # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the richest red channel values (e.g. >120).
        # Remember, pure white is bgr(255, 255, 255).
        # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))

        # Lane lines should be pure in color and have high red channel values
        # Bitwise AND operation to reduce noise and black-out any pixels that
        # don't appear to be nice, pure, solid colors (like white or yellow lane
        # lines.)
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))
        return self.lane_line_markings

    def histogram_peak(self):
        """
        Get the left and right peak of the histogram

        Return the x coordinate of the left histogram peak and the right histogram
        peak.
        """
        midpoint = int(self.histogram.shape[0] / 2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # (x coordinate of left peak, x coordinate of right peak)
        return leftx_base, rightx_base

    def overlay_lane_lines(self, plot=False, print_center_line=True):
        """
        Overlay lane lines on the original frame
        :param plot: Plot the lane lines if True
        :param print_center_line: Print the center line coordinates if True
        :return: Lane with overlay
        """
        # Generate an image to draw the lane lines on
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))]
        )
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.array([pts], dtype=np.int32), (0, 255, 0))

        # --- Draw the mid line (target path) in blue ---
        mid_fitx = (self.left_fitx + self.right_fitx) / 2
        mid_pts = np.array([np.transpose(np.vstack([mid_fitx, self.ploty]))], dtype=np.int32)
        cv2.polylines(color_warp, mid_pts, isClosed=False, color=(255, 0, 0), thickness=8)

        # For plotting: also draw the mid line on the warped frame
        warped_frame_with_mid = cv2.cvtColor(self.warped_frame, cv2.COLOR_GRAY2BGR) \
            if len(self.warped_frame.shape) == 2 or self.warped_frame.shape[2] == 1 \
            else self.warped_frame.copy()
        cv2.polylines(warped_frame_with_mid, mid_pts, isClosed=False, color=(255, 0, 0), thickness=8)

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp,
            self.inv_transformation_matrix,
            (self.orig_frame.shape[1], self.orig_frame.shape[0]),
        )

        # Combine the result with the original image
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        # Output center lane waypoints to terminal if requested
        if print_center_line:
            waypoints = list(zip(mid_fitx.astype(int), self.ploty.astype(int)))
            # print("Center lane waypoints (x, y):")
            # print(waypoints)

        if plot == True:
            # Plot the figures
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 15)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax3.imshow(cv2.cvtColor(warped_frame_with_mid, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay and Mid Line")
            ax3.set_title("Warped Frame With Mid Line")
            plt.show()
            images_dir = ensure_images_dir()
            fig_path = os.path.join(images_dir, "lane_overlay.png")
            figure.savefig(fig_path)

        return result

    def perspective_transform(self, frame=None, plot=False):
        """
        Perform the perspective transform.
        :param: frame Current frame
        :param: plot Plot the warped image if True
        :return: Bird's eye view of the current lane
        """
        if frame is None:
            frame = self.lane_line_markings

        # 使用 sign_block_points 在原圖座標系先行遮蔽道路中線（避免被當作車道線）
        try:
            if hasattr(self, "sign_block_points") and self.sign_block_points is not None:
                mask = np.zeros(self.orig_frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [self.sign_block_points.astype(np.int32)], 255)
                inv_mask = cv2.bitwise_not(mask)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    inv_mask_3 = cv2.merge([inv_mask, inv_mask, inv_mask])
                    frame = cv2.bitwise_and(frame, inv_mask_3)
                else:
                    # 單通道（binary/grayscale）
                    # 若 frame 尺寸與 orig_frame 不同，resize 遮罩以匹配
                    if frame.shape[:2] != mask.shape[:2]:
                        inv_mask_resized = cv2.resize(inv_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        frame = cv2.bitwise_and(frame, inv_mask_resized)
                    else:
                        frame = cv2.bitwise_and(frame, inv_mask)
        except Exception:
            # 遮罩非關鍵步驟，若失敗不影響主流程
            pass

        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points
        )

        # Calculate the inverse transformation matrix
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points
        )

        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
            frame,
            self.transformation_matrix,
            self.orig_image_size,
            flags=(cv2.INTER_LINEAR),
        )

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY
        )
        self.warped_frame = binary_warped

        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(
                warped_copy,
                np.int32([self.desired_roi_points]),
                True,
                (147, 20, 255),
                3,
            )

            # Display the image
            while 1:
                cv2.imshow("Warped Image", warped_plot)

                # Press any key to stop
                if cv2.waitKey(0):
                    break

            cv2.destroyAllWindows()

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        """
        Plot the region of interest on an image.
        :param: frame The current image frame
        :param: plot Plot the roi image if True
        """
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        # Overlay trapezoid on the frame
        this_image = cv2.polylines(
            frame, np.int32([self.roi_points]), True, (147, 20, 255), 3
        )

        # Display the image
        while 1:
            cv2.imshow("ROI Image", this_image)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

class LaneDetector:
    """
    車道檢測器 - 專為即時應用設計，保持狀態並重用計算
    """
    
    def __init__(self, frame_width, frame_height, plot_enabled=False):
        """
        初始化檢測器
        
        :param frame_width: 影像寬度
        :param frame_height: 影像高度
        :param plot_enabled: 是否啟用 plot 顯示
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lane_obj = None
        self.is_initialized = False
        self.plot_enabled = plot_enabled
        
        # 用於判斷是否使用滑動窗口的參數
        self.frames_since_detection = 0
        self.max_frames_without_detection = 5
        
    def initialize(self, first_frame):
        """
        使用第一幀初始化車道檢測器
        
        :param first_frame: 第一幀影像
        """
        # 調整影像大小（如果需要）
        if first_frame.shape[1] != self.frame_width or first_frame.shape[0] != self.frame_height:
            first_frame = cv2.resize(first_frame, (self.frame_width, self.frame_height))
        
        self.lane_obj = Lane(orig_frame=first_frame)
        
        # 執行初始檢測
        lane_line_markings = self.lane_obj.get_line_markings()
        warped_frame = self.lane_obj.perspective_transform()
        histogram = self.lane_obj.calculate_histogram(plot=self.plot_enabled)
        
        # 使用滑動窗口進行第一次檢測
        left_fit, right_fit = self.lane_obj.get_lane_line_indices_sliding_windows(plot=self.plot_enabled)
        
        if left_fit is not None and right_fit is not None:
            self.is_initialized = True
            self.frames_since_detection = 0
            # print("車道檢測器初始化成功")
        else:
            pass
            # print("車道檢測器初始化失敗")
            
        return self.is_initialized
    
    def enable_plot(self):
        """
        啟用 plot 顯示
        """
        self.plot_enabled = True
    
    def disable_plot(self):
        """
        關閉 plot 顯示
        """
        self.plot_enabled = False
    
    def set_plot(self, enabled):
        """
        設定 plot 顯示狀態
        
        :param enabled: True 為啟用，False 為關閉
        """
        self.plot_enabled = enabled
    
    def process_frame(self, frame, force_sliding_window=False, show_real_time=False):
        """
        處理單一幀影像（即時使用）
        
        :param frame: 輸入影像 (numpy.ndarray)
        :param force_sliding_window: 強制使用滑動窗口 (bool)
        :param show_real_time: 是否在回傳影像上繪製額外資訊 (bool)
        :return: 一個包含三個元素的元組 (tuple):
            - **processed_frame (numpy.ndarray)**: 處理後的影像。
              如果檢測成功，則為帶有車道覆蓋和資訊的影像；
              如果失敗，則為原始影像。
            - **success (bool)**: 如果成功檢測到車道線，則為 ``True``，否則為 ``False``。
            - **lane_info (dict or None)**: 如果檢測成功，則為包含車道資訊的字典，否則為 ``None``。
              字典包含以下鍵值:
                - ``'left_curvature'`` (float): 左車道線的曲率半徑（米）。
                - ``'right_curvature'`` (float): 右車道線的曲率半徑（米）。
                - ``'center_offset'`` (float): 車輛中心相對於車道中心的偏移量（厘米）。
                - ``'turn_direction'`` (str): 建議的轉向方向 ('left', 'right', 'straight')。
                - ``'steer_deg'`` (float): 建議的轉向角度（度），左轉為正，右轉為負。
                - ``'signed_radius_m'`` (float): 帶正負號的車道中心曲率半徑（米）。左轉為正，右轉為負，直線為 inf。
                - ``'detection_method'`` (str): 使用的檢測方法 ('sliding_window' 或 'previous_window')。
        """
        if not self.is_initialized:
            success = self.initialize(frame)
            if not success:
                return frame, False, None
        
        # 調整影像大小
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # 更新 Lane object 的當前幀
        self.lane_obj.orig_frame = frame.copy()
        
        try:
            # 取得車道線標記
            lane_line_markings = self.lane_obj.get_line_markings()
            warped_frame = self.lane_obj.perspective_transform(plot=self.plot_enabled)
            
            # 決定使用哪種檢測方法
            use_sliding_window = (
                force_sliding_window or 
                self.frames_since_detection >= self.max_frames_without_detection or
                self.lane_obj.left_fit is None or 
                self.lane_obj.right_fit is None
            )
            
            # if use_sliding_window:
            #     # 使用滑動窗口重新檢測
            #     histogram = self.lane_obj.calculate_histogram(plot=self.plot_enabled)
            #     left_fit, right_fit = self.lane_obj.get_lane_line_indices_sliding_windows(plot=self.plot_enabled)
            #     self.frames_since_detection = 0
            # else:
            #     # 使用前一幀的結果進行快速檢測
            #     left_fit = self.lane_obj.left_fit
            #     right_fit = self.lane_obj.right_fit
            #     self.lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=self.plot_enabled)
            
            histogram = self.lane_obj.calculate_histogram(plot=self.plot_enabled)
            left_fit, right_fit = self.lane_obj.get_lane_line_indices_sliding_windows(plot=self.plot_enabled)
            self.lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=self.plot_enabled)
            
            # 檢查檢測結果是否有效
            if left_fit is not None and right_fit is not None:
                # 疊加車道線
                frame_with_lanes = self.lane_obj.overlay_lane_lines(plot=self.plot_enabled)
                
                # 計算曲率和偏移
                self.lane_obj.calculate_curvature(print_to_terminal=False)
                self.lane_obj.calculate_car_position(print_to_terminal=False)
                
                # 顯示資訊
                if show_real_time:
                    final_frame = self.lane_obj.display_curvature_offset(
                        frame=frame_with_lanes, plot=False
                    )
                else:
                    final_frame = frame_with_lanes

                # 計算轉向建議
                direction, steer_deg, signed_R = self.lane_obj.compute_turn_command()
                
                # 準備車道資訊
                lane_info = {
                    'left_curvature': self.lane_obj.left_curvem,
                    'right_curvature': self.lane_obj.right_curvem,
                    'center_offset': self.lane_obj.center_offset,
                    'turn_direction': direction,
                    'steer_deg': steer_deg,
                    'signed_radius_m': signed_R,
                    'detection_method': 'sliding_window' if use_sliding_window else 'previous_window'
                }
                
                return final_frame, True, lane_info
            
            else:
                # 檢測失敗
                self.frames_since_detection += 1
                return frame, False, None
                
        except Exception as e:
            print(f"車道檢測錯誤: {e}")
            self.frames_since_detection += 1
            return frame, False, None
    
    def reset(self):
        """
        重置檢測器狀態
        """
        self.lane_obj = None
        self.is_initialized = False
        self.frames_since_detection = 0
        print("車道檢測器已重置")
    
    def get_lane_center_point(self, y_position=None):
        """
        取得指定高度的車道中心點 X 座標
        
        :param y_position: Y座標位置（None為底部）
        :return: 車道中心點的 X 座標
        """
        if not self.is_initialized or self.lane_obj.left_fit is None or self.lane_obj.right_fit is None:
            return None
            
        if y_position is None:
            y_position = self.frame_height - 1
            
        # 計算該高度的左右車道線位置
        left_x = (self.lane_obj.left_fit[0] * y_position**2 + 
                 self.lane_obj.left_fit[1] * y_position + 
                 self.lane_obj.left_fit[2])
        right_x = (self.lane_obj.right_fit[0] * y_position**2 + 
                  self.lane_obj.right_fit[1] * y_position + 
                  self.lane_obj.right_fit[2])
        
        return (left_x + right_x) / 2

def process_one_frame(frame, plot=False, show_real_time=False):
    """
    即時影片車道檢測
    
    :param frame: 輸入影像
    :param plot: 是否顯示 plot
    """
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # 建立車道檢測器
    detector = LaneDetector(frame_width, frame_height, plot_enabled=plot)
    
    # print("開始即時車道檢測")

    # 處理幀
    result_frame, success, lane_info = detector.process_frame(frame, force_sliding_window=False, show_real_time=show_real_time)
    
    # 顯示檢測狀態
    if show_real_time:
        status_text = "lane detection" if success else "lane detection failed"
        cv2.putText(result_frame, status_text, (350, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0) if success else (0, 0, 255), 2)
        
        if success and lane_info:
            # 顯示車道中心點
            center_x = detector.get_lane_center_point()
            if center_x:
                cv2.circle(result_frame, (int(center_x), frame_height - 50), 
                        10, (255, 0, 0), -1)
        
        # 儲存/顯示結果
        cv2.imshow('Real-time Lane Detection', result_frame)
    
    return result_frame, success, lane_info

def ensure_images_dir():
    """
    Ensure the 'images' directory exists.
    """
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

if __name__ == "__main__":
    # main()
    # frame = cv2.imread("british airways landing-short-00.00.05.773.jpeg")
    frame = cv2.imread("front_20250815_145515_567427.jpg")
    result_frame, success, lane_info = process_one_frame(frame, plot=False, show_real_time=True)
    print(lane_info)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
