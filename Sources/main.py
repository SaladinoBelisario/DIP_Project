import random

import cv2
from line import Line
from scipy import ndimage
import numpy as np
import filtros as flt
from definitions import *

images = {}
left_line = Line()
right_line = Line()

std_x, std_y = (1024, 1024)

# Create a trapezoid in where we search for lanes
x, y = std_x, std_y
tl = [390, 590]
tr = [690, 590]
br = [950, 990]
bl = [100, 990]
corner_points_array = np.float32([tl, tr, br, bl])

# Create an array with the parameters (the dimensions) required to build the matrix
imgTl = [0, 0]
imgTr = [x, 0]
imgBr = [x, y]
imgBl = [0, y]
img_params = np.float32([imgTl, imgTr, imgBr, imgBl])


def prepare_sample_images(folder):
    file_selected = [f for f in os.listdir(folder)]
    for filename in file_selected:
        img = cv2.imread(os.path.join(folder, filename))
        images[filename] = img


def laplacian_processing(gray_img, tam_kernel = 3):
    lapl = flt.laplacian_filter(tam_kernel) * np.transpose(flt.gaussian_filter(tam_kernel))
    return ndimage.convolve(gray_img, lapl)


def blur_processing(gray_img, tam_kernel = 32):
    blur = flt.gaussian_blur(tam_kernel)
    return ndimage.convolve(gray_img, blur)


def hsl_threshold(rgb_img, min_thres = 200, max_thres = 255):
    hsl_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    l_channel = hsl_img[:, :, 1]
    binary = np.zeros_like(l_channel)
    binary[(l_channel > min_thres) & (l_channel <= max_thres)] = 1
    return binary


def pers_transform(img, src, dst):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    return warped, M, Minv


def window_search(binary_warped):
    # Take a histogram of the bottom half of the image
    bottom_half_y = binary_warped.shape[0] / 2
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis = 0)
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
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
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
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # Generate black image and colour lane lines
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
    
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1, 1, 0), thickness = 5)
    cv2.polylines(out_img, [left], False, (1, 1, 0), thickness = 5)
    
    return left_lane_inds, right_lane_inds, out_img


def margin_search(binary_warped):
    # Performs window search on subsequent frame, given previous frame.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30
    
    left_lane_inds = ((nonzerox > (left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                                   left_line.current_fit[2] - margin)) & (nonzerox < (
            left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
            left_line.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                                    right_line.current_fit[2] - margin)) & (nonzerox < (
            right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
            right_line.current_fit[2] + margin)))
    
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
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # Generate a blank image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.intc([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.intc([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
    
    # Draw polyline on image
    right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
    left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
    cv2.polylines(out_img, [right], False, (1, 1, 0), thickness = 5)
    cv2.polylines(out_img, [left], False, (1, 1, 0), thickness = 5)
    
    return left_lane_inds, right_lane_inds, out_img


def validate_lane_update(img, left_lane_inds, right_lane_inds):
    # Checks if detected lanes are good enough before updating
    img_size = (img.shape[1], img.shape[0])
    
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    left_line_allx = nonzerox[left_lane_inds]
    left_line_ally = nonzeroy[left_lane_inds]
    right_line_allx = nonzerox[right_lane_inds]
    right_line_ally = nonzeroy[right_lane_inds]
    
    # Discard lane detections that have very little points,
    # as they tend to have unstable results in most cases
    if len(left_line_allx) <= 1800 or len(right_line_allx) <= 1800:
        left_line.detected = False
        right_line.detected = False
        return
    
    left_x_mean = np.mean(left_line_allx, axis = 0)
    right_x_mean = np.mean(right_line_allx, axis = 0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    
    # Discard the detections if lanes are not in their repective half of their screens
    # mid_x = np.floor(img_size[0])
    # if left_x_mean > mid_x or right_x_mean < mid_x:
    #     left_line.detected = False
    #     right_line.detected = False
    #     return
    
    # Discard the detections if the lane width is too large or too small
    if lane_width < 256 or lane_width > 768:
        left_line.detected = False
        right_line.detected = False
        return
    
    # If this is the first detection or
    # the detection is within the margin of the averaged n last lines
    if left_line.bestx is None or np.abs(np.subtract(left_line.bestx, np.mean(left_line_allx, axis = 0))) < 100:
        left_line.update_lane(left_line_ally, left_line_allx)
        left_line.detected = True
    else:
        left_line.detected = False
    if right_line.bestx is None or np.abs(np.subtract(right_line.bestx, np.mean(right_line_allx, axis = 0))) < 100:
        right_line.update_lane(right_line_ally, right_line_allx)
        right_line.detected = True
    else:
        right_line.detected = False


def find_lanes(img):
    if left_line.detected and right_line.detected:  # Perform margin search if exists prior success.
        # Margin Search
        left_lane_inds, right_lane_inds, out_img = margin_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_inds, right_lane_inds)
    
    else:  # Perform a full window search if no prior successful detections.
        # Window Search
        left_lane_inds, right_lane_inds, out_img = window_search(img)
        # Update the lane detections
        validate_lane_update(img, left_lane_inds, right_lane_inds)
    return out_img


def draw_lane(undist, img, Minv):
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.stack((warp_zero, warp_zero, warp_zero), axis = -1)
    
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    
    if left_fit is not None and right_fit is not None:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (64, 224, 208))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.6, 0)
        return result
    return undist


def process_random_sample(intermidiate_process = 0):
    img_name = random.sample(images.keys(), 1)
    img = images[img_name[0]]
    img = cv2.resize(img, (std_x, std_y))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = blur_processing(gray, 16)
    cv2.imshow("Original", img)
    
    # Compute and return the transformation matrix
    img_transformed, matrix, matrixInv = pers_transform(img, corner_points_array, img_params)
    # We transform the blurred image to use it with laplacian filter
    img_g, _, _ = pers_transform(blurred, corner_points_array, img_params)
    # Threshold to the transformed RGB image
    hsl_bin = hsl_threshold(img_transformed, min_thres = 165)
    # Laplace filter to gray transformed image
    lpl_bin = laplacian_processing(img_g)
    # Combine the filters
    combined = np.zeros_like(hsl_bin)
    combined[(hsl_bin == 1) | (lpl_bin == 1)] = 1
    
    # Find Lanes
    output_img = find_lanes(combined)
    
    if intermidiate_process:
        cv2.imshow("Grey_blurred", blurred)
        cv2.imshow("transformation", img_transformed)
        cv2.imshow("grey_transform", img_g)
        cv2.imshow("h_bin", hsl_bin * 255)
        cv2.imshow("l_bin", lpl_bin * 255)
        cv2.imshow("draw lane", output_img)
    
    # Draw lanes on image
    lane_img = draw_lane(img, combined, matrixInv)
    cv2.imshow("lanes on image", lane_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


prepare_sample_images(PICTURES_DIR)
process_random_sample(1)
