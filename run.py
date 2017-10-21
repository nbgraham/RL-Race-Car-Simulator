import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import cv2

from myplot import plotRewards

from luc/luc_model import Model, create_nn
from lucluc_preprocessing import compute_state

gamma = 0.99
N = 1001
eps_coeff=0.5 # initially 0.5

global_n = 0

def main():
    try:
        with open("episode_file.txt", "r") as episode_file:
            continue_from = episode_file.read()
            run_simulator(int(continue_from)

        with open("episode_file.txt", "w") as episode_file:
            episode_file.write("0")
    except KeyboardInterrupt:
        if global_n != -1:
            with open("episode_file.txt", "w") as episode_file:
                episode_file.write(str(global_n))


def run_simulator(continue_from=0):
    env = gym.make('CarRacing-v0')
    env = wrappers.Monitor(env, 'monitor-folder', force=True)

    model = Model(env)

    totalrewards = np.empty(N)
    if continue_from > 0:
        f = open('rewards.npy', 'rb')
        totalrewards = np.load(f)
    costs = np.empty(N)

    plt.ion()
    plt.show()

    for n in range(continue_from,N):
        global_n = n
        eps = eps_coeff/np.sqrt(n+1 + 900)
        totalreward, iters = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 1 == 0:
          print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
        if n % 10 == 0:
            avg_rewards = [totalrewards[max(0, i-100):(i+1)].mean() for i in range(n)]
            plt.plot(avg_rewards)
            plt.draw()
            plt.pause(0.001)

            model.model.save('race-car.h5')

            with open('rewards.npy', 'wb') as out_reward_file:
                np.save(out_reward_file, totalrewards)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())


    plotRewards("Luc Prieur's NN", totalrewards, N/10)


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


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
        steering = float(output_value)/4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gas = float(output_value)/3 # 33%
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2 # 50% brakes
    else:
        print("error")

    white = np.ones((round(brake * 100), 10))
    black = np.zeros((round(100 - brake * 100), 10))
    brake_display = np.concatenate((black, white))*255

    white = np.ones((round(gas * 100), 10))
    black = np.zeros((round(100 - gas * 100), 10))
    gas_display = np.concatenate((black, white))*255

    control_display = np.concatenate((brake_display, gas_display), axis=1)

    cv2.imshow('controls', control_display)
    cv2.waitKey(1)

    return [steering, gas, brake]


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0

    state = compute_state(observation)

    while not done:
        state = compute_state(observation)

        argmax_qval, qval = model.sample_action(state, eps)
        action = convert_argmax_qval_to_env_action(argmax_qval)
        observation, reward, done, info = env.step(action)

        prev_state = state

        state = compute_state(observation)

        # update the model
        # standard Q learning TD(0)
        next_qval = model.predict(state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1

        if iters > 1500:
            print("This episode is stuck")
            break

    return totalreward, iters


if __name__ == "__main__":
    main()
