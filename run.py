import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import cv2
from os import path

from myplot import plotRewards

from luc.luc_model import Model
from luc.luc_preprocessing import compute_state
from luc.eps import get_eps

global_n = 0
def main():
    N = 1001
    name = ""
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        print("Must give method name")
        return
    if len(sys.argv) > 2:
        N = sys.argv[2]

    episode_filename = "episode_file_" + name + ".txt"
    model_filename = "race_car_" + name + ".h5"
    reward_filename = "rewards_" + name + ".npy"
    try:
        continue_from=0
        if path.exists(episode_filename):
            with open(episode_filename, "r") as episode_file:
                continue_from = int(episode_file.read())
        run_simulator(continue_from, N, model_filename, reward_filename)

        with open(episode_filename, "w") as episode_file:
            episode_file.write("0")
    except KeyboardInterrupt:
        if global_n != -1:
            with open(episode_filename, "w") as episode_file:
                episode_file.write(str(global_n))


def run_simulator(continue_from, N, model_filename, reward_filename):
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
        eps = get_eps(n)
        totalreward, iters = play_one(env, model, eps)
        totalrewards[n] = totalreward

        print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

        if n % 10 == 0:
            avg_rewards = [totalrewards[max(0, i-100):(i+1)].mean() for i in range(n)]
            plt.plot(avg_rewards)
            plt.draw()
            plt.pause(0.001)

            model.model.save(model_filename)

            with open(reward_filename, 'wb') as out_reward_file:
                np.save(out_reward_file, totalrewards)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total reward:", totalrewards.sum())


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def play_one(env, model, eps):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0
    reward = 0

    while not done:
        state = compute_state(observation)

        action = model.get_action(state, eps, reward)

        observation, reward, done, info = env.step(action)

        totalreward += reward
        iters += 1

        if iters > 1500:
            print("This episode is stuck")
            break

    return totalreward, iters


if __name__ == "__main__":
    main()
