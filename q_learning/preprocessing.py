import cv2
import numpy as np


def compute_state(observation):
    bottom_black_bar_bw, upper_field_bw, car_field_t = transform(observation)
    dashboard_values = compute_steering_speed_gyro_abs(bottom_black_bar_bw)
    state = np.concatenate((
        np.array([dashboard_values]).reshape(1,-1).flatten(),
        upper_field_bw.reshape(1,-1).flatten(),
        car_field_t),
    axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1
    return state


def transform(rgb_pixel_matrix):
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    # bottom_black_bar is the section of the screen with steering, speed, abs and gyro information.
    # we crop off the digits on the right as they are illigible, even for ml.
    # since color is irrelavent, we grayscale it.
    bottom_black_bar = rgb_pixel_matrix[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)

    upper_field = rgb_pixel_matrix[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
    upper_field_bw = upper_field_bw.astype('float')/255

    car_field = rgb_pixel_matrix[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    car_field_t = [car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255]

    return bottom_black_bar_bw, upper_field_bw, car_field_t


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

def speed_cnn(observation):
    bottom_black_bar = observation[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation=cv2.INTER_NEAREST)

    right_steering = bottom_black_bar_bw[6, 36:46].mean() / 255
    left_steering = bottom_black_bar_bw[6, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = bottom_black_bar_bw[6, 46:60].mean() / 255
    right_gyro = bottom_black_bar_bw[6, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = bottom_black_bar_bw[:, 0][:-2].mean() / 255
    abs1 = bottom_black_bar_bw[:, 6][:-2].mean() / 255
    abs2 = bottom_black_bar_bw[:, 8][:-2].mean() / 255
    abs3 = bottom_black_bar_bw[:, 10][:-2].mean() / 255
    abs4 = bottom_black_bar_bw[:, 12][:-2].mean() / 255

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]
