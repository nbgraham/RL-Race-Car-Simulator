import numpy as np

from old_models.preprocessing import get_dashboard_values, cropped_grayscale_car_road

vector_size = (90,90,1)

def compute_state(obs):
    values = get_dashboard_values(obs)
    [steering, speed, gyro, abs1, abs2, abs3, abs4] = [255*val for val in values]

    track_view = cropped_grayscale_car_road(obs)

    zs = np.zeros((81,1))
    ninezs = [zs]*9
    track_with_zeroes = [track_view]
    track_with_zeroes.extend(ninezs)
    track_view = np.hstack(track_with_zeroes)
    track_view.resize((90,90))

    track_view[81:,0:27] = steering
    track_view[81:,27:54] = speed
    track_view[81:,54:81] = gyro

    track_view[0:27,81:] = abs4
    track_view[27:54,81:] = abs3
    track_view[54:81,81:] = abs2

    track_view[81:,81:] = abs1

    return track_view