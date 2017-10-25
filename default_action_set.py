import numpy as np

def convert_argmax_qval_to_env_action(output_value):
    # we reduce the action space to 15 values.  9 for steering, 6 for gas/brake.
    # to reduce the action space, gas and brake cannot be applied at the same time.
    # as well, steering input and gas/brake cannot be applied at the same time.
    # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

    gas = 0.0
    brake = 0.0
    steering = 0.0

    # output value ranges from 0 to 10

    if output_value <= 8:
        # steering. brake and gas are zero.
        output_value -= 4
        steering = float(output_value) / 4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gas = float(output_value) / 3  # 33%
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value) / 2  # 50% brakes
    else:
        print("error")

    white = np.ones((round(brake * 100), 10))
    black = np.zeros((round(100 - brake * 100), 10))
    brake_display = np.concatenate((black, white)) * 255

    white = np.ones((round(gas * 100), 10))
    black = np.zeros((round(100 - gas * 100), 10))
    gas_display = np.concatenate((black, white)) * 255

    return [steering, gas, brake]

default_action_set = [convert_argmax_qval_to_env_action(i) for i in range(11)]