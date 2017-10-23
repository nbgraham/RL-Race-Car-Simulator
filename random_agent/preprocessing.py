import cv2
import numpy as np


def compute_state(observation):
    return 0


# this function uses the bottom black bar of the screen and extracts steering setting, speed and gyro data
def compute_steering_speed_gyro_abs(black_bar_pixel_matrix):
    right_steering = black_bar_pixel_matrix[6, 36:46].mean()/255
    left_steering = black_bar_pixel_matrix[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2

    left_gyro = black_bar_pixel_matrix[6, 46:60].mean()/255
    right_gyro = black_bar_pixel_matrix[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2

    speed = black_bar_pixel_matrix[:, 0][:-2].mean()/255
    abs1 = black_bar_pixel_matrix[:, 6][:-2].mean()/255
    abs2 = black_bar_pixel_matrix[:, 8][:-2].mean()/255
    abs3 = black_bar_pixel_matrix[:, 10][:-2].mean()/255
    abs4 = black_bar_pixel_matrix[:, 12][:-2].mean()/255

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

#this function gets above the black bar for the cnn state input
def above_bar_cnn(observation):
    above_bar = observation[:84,6:90]
    img = cv2.cvtColor(above_bar, cv2.COLOR_RGB2GRAY)
    #if pixel > 120 set to white else black
    # above_bar_bw = cv2.threshold(img,120,255,cv2.THRESH_BINARY)[1]
    # above_bar_bw = above_bar_bw.astype('float')/255
    above_bar_bw = img.astype('float')/255
    return np.resize(above_bar_bw,(84,84,1))
