import numpy as np

def focus_middle(observation):
    car_road_gray = cropped_grayscale_car_road(observation)
    coarse = rebin(car_road_gray, (27,27))

    focused_middle = np.hstack([coarse[:,0:9], car_road_gray[::3,27:53],coarse[:,18:27]])

    return focused_middle

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
