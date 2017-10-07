import numpy as np
import cv2


# https://gym.openai.com/evaluations/eval_BPzPoiBtQOCj8yItyHLhmg/
def bottom_bar(observation):
    bottom_black_bar = observation[84:, 12:]
    # convert to gray
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    #if pixel value > 1 set to white else black
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    # resize to 84 columns (width) * 12 rows (height) using nearest neighbor interpolation
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)

    return bottom_black_bar_bw

#https://gym.openai.com/evaluations/eval_BPzPoiBtQOCj8yItyHLhmg/
def above_bar(observation):
    #getting above the black bar and cropping from the sides (originally 96, now 84, giving a square)
    above_bar = observation[:84,6:90]
    #convert to gray
    img = cv2.cvtColor(above_bar, cv2.COLOR_RGB2GRAY)
    #if pixel > 120 set to white else black
    above_bar_bw = cv2.threshold(img,120,255,cv2.THRESH_BINARY)[1]
    #resize to 10 columns (width) * 10 rows (height) using nearest neighbor interpolation
    above_bar_bw = cv2.resize(above_bar_bw,(10,10),interpolation=cv2.INTER_NEAREST)
    #set to float values between 0 and 1
    above_bar_bw = above_bar_bw.astype('float')/255

    return above_bar_bw

def car_field(observation):
    #getting location of car (columns, rows)
    car_field = observation[66:78, 43:53]
    #convert to gray
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    #if pixel > 80 set to white else black
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    #print('car field bw\n', car_field_bw)
    #average of columns 3, 4, 5 and 6 (the columns where the car is in the car field)
    car_field_t = [car_field_bw[:, 3].mean() / 255, car_field_bw[:, 4].mean() / 255, car_field_bw[:, 5].mean() / 255,
                   car_field_bw[:, 6].mean() / 255]

    return car_field_t

def compute_state(observation):
    above_bar_bw = above_bar(observation)
    car_field_t = car_field(observation)
    dashboard_values = get_dashboard_values(observation)

    #print('car field t\n', car_field_t)
    #print('above bar\n', above_bar_bw)
    #print('dashboard values (steering, speed, gyro, abs1, abs2,abs3, abs4)\n', dashboard_values)

    #turn into 1d array for neural net
    state = np.concatenate((
        np.array([dashboard_values]).reshape(1,-1).flatten(),
        above_bar_bw.reshape(1,-1).flatten(),
        car_field_t),axis=0)

    #print('state\n', state)

    return state

#https://gym.openai.com/evaluations/eval_BPzPoiBtQOCj8yItyHLhmg/
def compute_steering_speed_gyro_abs(bottom_black_bar_gray_array):
    right_steering = bottom_black_bar_gray_array[6, 36:46].mean()/255
    left_steering = bottom_black_bar_gray_array[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2

    left_gyro = bottom_black_bar_gray_array[6, 46:60].mean()/255
    right_gyro = bottom_black_bar_gray_array[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2

    speed = bottom_black_bar_gray_array[:, 0][:-2].mean()/255
    abs1 = bottom_black_bar_gray_array[:, 6][:-2].mean()/255
    abs2 = bottom_black_bar_gray_array[:, 8][:-2].mean()/255
    abs3 = bottom_black_bar_gray_array[:, 10][:-2].mean()/255
    abs4 = bottom_black_bar_gray_array[:, 12][:-2].mean()/255

    return [steering, speed, gyro, abs1, abs2, abs3, abs4]


def get_dashboard_values(observation):
    bottom_black_bar_gray_array = bottom_bar(observation)
    dashboard_values = compute_steering_speed_gyro_abs(bottom_black_bar_gray_array)
    return dashboard_values


def focus_middle(observation):
    car_road_gray = cropped_grayscale_car_road(observation)
    coarse = rebin(car_road_gray, (27,27))

    focused_middle = np.hstack([coarse[:,0:9], car_road_gray[::3,27:53],coarse[:,18:27]])

    return focused_middle


def coarse(observation):
    car_road_gray = cropped_grayscale_car_road(observation)
    coarse = rebin(car_road_gray, (9,9))
    return coarse

def fine_car(observation):
    car_road_gray = cropped_grayscale_car_road(observation)
    return car_road_gray[54:81,27:54]


def focus_car(observation):
    car_road_gray = cropped_grayscale_car_road(observation)
    coarse = rebin(car_road_gray, (9,9))

    # remove focused area
    coarse[6:9,3:6] = -1
    coarse_list = coarse[coarse>=0].ravel()

    fine_list = car_road_gray[54:81,27:54].ravel()

    return np.hstack([coarse_list, fine_list])


def cropped_grayscale_car_road(observation):
    cropped = observation[:81, 8:89]
    gray = rgb2gray(cropped)
    car_road_gray = just_car_road(gray)

    return car_road_gray


def just_car_road(gray):
    gray[gray<60] = 0
    gray[btw(gray,60,63)] = -61
    gray[btw(gray,63,100)] = 0
    gray[btw(gray,100,120)] = -100
    gray[gray>=120] = 0

    gray[gray==-61]=255
    gray[gray==-100]=127

    return gray


def btw(x,a,b):
    lower = a<=x
    upper = x<b
    combined = lower * upper
    return combined


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
