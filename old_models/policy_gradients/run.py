import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import cv2

from model import Model, create_nn
from preprocessing import compute_state

gamma = 0.99
N = 500
eps_coeff=0.5 # first run was 0.5
learning_rate = 0.01

action_set = np.array([
#nothing
[0,0,0],
#steering
[-1.0,0,0],
[-0.5,0,0],
[0,0,0],
[0.5,0,0],
[1.0,0,0],
#gas
[0,0.3,0],
[0,0.5,0],
[0,0.8,0],
#brake
[0,0,0.5],
[0,0,0.8]
])


def main(softmax):
    modelfile = 'race-car-prob.h5' if softmax else 'race-car.h5'
    rewardfile = 'rewards-prob.npy' if softmax else 'rewards.npy'

    env = gym.make('CarRacing-v0')
    env = wrappers.Monitor(env, 'monitor-folder', force=True)

    model = Model(env)

    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps_stretch=5
        eps = np.sqrt(eps_stretch/(n+eps_stretch))
        totalreward, iters = play_one(env, model, eps, gamma)
        totalrewards[n] = totalreward
        if n % 1 == 0:
          print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
        if n % 10 == 0:
            model.model.save(modelfile)

            with open(rewardfile, 'wb') as out_reward_file:
                np.save(out_reward_file, totalrewards)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def get_env_action(argmax_pi):
    action_selector = np.zeros((1,11))
    action_selector[0,argmax_pi] = 1
    action = np.dot(action_selector,action_set).ravel()
    return action, action_selector


def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0

    reward_decay = 0.8
    residual_reward = 0
    while not done:
        state = compute_state(observation)

        argmax_pi, pi = model.sample_action(state, eps)
        action, action_selector = get_env_action(argmax_pi)
        observation, reward, done, info = env.step(action)

        residual_reward = reward_decay*residual_reward + reward

        prev_state = state
        state = compute_state(observation)

        # update the model
        target = compute_target(residual_reward, action_selector, pi)
        model.update(prev_state, target)
        totalreward += reward
        iters += 1

        if iters > 1500:
            print("This episode is stuck")
            break

    return totalreward, iters


def compute_target(residual_reward, action_selector, pi_vals):
    scalar = residual_reward*2*learning_rate
    target = pi_vals + scalar*action_selector
    return target


if __name__ == "__main__":
    main(False)
