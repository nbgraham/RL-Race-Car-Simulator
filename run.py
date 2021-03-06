import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from os import path

from convolutional.model import Model
from convolutional.preprocessing import compute_state, vector_size as input_size
from default_action_set import default_action_set

global_episode_n = 0


def main():
    name, N = get_params()
    episode_filename = "saved_episodes/episode_file_" + name + ".txt"

    try:
        continue_from = get_last_episode(episode_filename)
        run_simulator(continue_from, N, name)

        save_last_episode(episode_filename, 0)
    except KeyboardInterrupt:
        if global_episode_n != 0:
            save_last_episode(episode_filename, global_episode_n)


def get_params():
    name = ""
    N = 1001
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        print("Must give method name")
        return
    if len(sys.argv) > 2:
        N = int(sys.argv[2])

    return name, N


def get_last_episode(episode_filename):
    continue_from=0
    if path.exists(episode_filename):
        with open(episode_filename, "r") as episode_file:
            continue_from = int(episode_file.read())
    return continue_from


def save_last_episode(episode_filename, episode_n):
    with open(episode_filename, "w") as episode_file:
        episode_file.write(str(episode_n))


def run_simulator(continue_from, N, name):
    model_filename = "models/race_car_" + name + ".h5"
    reward_filename = "rewards/rewards_" + name + ".npy"
    loss_filename = "losses/loss_" + name + ".npy"
    monitor_foldername = "monitors/monitor_folder_" + name

    env = gym.make('CarRacing-v0')
    env = wrappers.Monitor(env, monitor_foldername, force=True)

    model = Model(env, name, input_size, default_action_set)

    totallosses = np.empty(N)
    if continue_from > 0:
        f = open(loss_filename, 'rb')
        totallosses = np.load(f)
        totallosses.resize(N)

    totalrewards = np.empty(N)
    if continue_from > 0:
        f = open(reward_filename, 'rb')
        totalrewards = np.load(f)
        totalrewards.resize(N)

    plt.ion()
    plt.title(name)
    plt.show()

    for n in range(continue_from,N):
        global_episode_n = n
        eps = Model.get_action_selection_parameter(n, N)

        totalreward, iters, totalloss = play_one(env, model, eps)
        totalrewards[n] = totalreward
        totallosses[n] = totalloss

        print("episode:", n, "eps:", round(eps,3), "iters", iters, "total loss:", round(totalloss), "avg loss (last 100):", round(totallosses[max(0, n-100):(n+1)].mean()), "total reward:", round(totalreward), "avg reward (last 100):", round(totalrewards[max(0, n-100):(n+1)].mean()))

        if n % 100 == 0:
            model.model.save(model_filename)
            with open(reward_filename, 'wb') as out_reward_file:
                np.save(out_reward_file, totalrewards)
            with open(loss_filename, 'wb') as out_loss_file:
                np.save(out_loss_file, totallosses)

            avg_rewards = [totalrewards[max(0, i-100):(i+1)].mean() for i in range(n)]
            plt.plot(avg_rewards)
            plt.draw()
            plt.pause(0.001)

    model.model.save(model_filename)
    with open(reward_filename, 'wb') as out_reward_file:
        np.save(out_reward_file, totalrewards)
    with open(loss_filename, 'wb') as out_loss_file:
        np.save(out_loss_file, totallosses)

    global_episode_n = 0
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total reward:", totalrewards.sum())

    plot_running_avg(totalrewards)


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
    totalreward = 0
    totalloss = 0
    iters = 0

    reward = 0


    while not done:
        state = compute_state(observation)

        action, loss = model.get_action(state, eps, reward)

        observation, reward, done, info = env.step(action)

        totalreward += reward
        totalloss += loss
        iters += 1

        if iters > 1500:
            print("This episode is stuck")
            break

    return totalreward, iters, totalloss


if __name__ == "__main__":
    main()
