import gym
import numpy as np
import json
import cv2
import math
import pickle

import myplot
import preprocessing as pre

render = True # Does't work if false, observations are wrong

max_time_steps = 2000
action_time_steps = 5
batch_size = 10

target_reward_per_frame = 1.5

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
[0,0.5,0],
[0,0.8,0],
#brake
[0,0,0.5],
[0,0,0.8]
])


def main(k):
    f = open('road_means_' + str(k) + '.npy','rb')
    road_means = np.load(f)
    train(100, road_means, 0.001, 0.99)


def train(n_episodes, road_means, alpha, discount, load=False, save=True):
    env = gym.make('CarRacing-v0')

    state_action_value = {}
    if load:
        f = open('q_vals.pkl', 'rb')
        state_action_value = pickle.load(f)

    rewards=[]
    for i_episode in range(n_episodes):
        eps = abs(i_episode - n_episodes)/n_episodes

        observation = env.reset()
        sum_reward = 0
        interval_reward = 0
        prev_state = None
        prev_action = None

        for t in range(max_time_steps):
            if render: env.render()

            if t % action_time_steps == 0:
                state = get_state(observation, road_means)
                action = get_env_action(state, state_action_value, eps) # [steering, gas, brake]

                if prev_state is not None and prev_action is not None:
                    if str(prev_state) not in state_action_value:
                        state_action_value[str(prev_state)] = {}
                    state_action_value[str(prev_state)][str(prev_action)] = q_update(prev_state, state, prev_action, state_action_value, alpha, interval_reward, discount)

                prev_state = state
                prev_action = action

                # cv2.imshow('road state', road_means[state[2]])
                # cv2.waitKey(1)

                interval_reward = 0


            observation, reward, done, info = env.step(action) # observation is 96x96x3

            interval_reward += reward
            sum_reward += reward

            if done or t==max_time_steps-1:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break

    with open('rewards/rewards','w') as out_reward_file:
        json.dump(rewards, out_reward_file)

    if save:
        with open('q_vals.pkl','wb') as out_q_val_file:
            pickle.dump(state_action_value, out_q_val_file, protocol=pickle.HIGHEST_PROTOCOL)

    myplot.plotRewards("K-means Q learner", rewards, int(n_episodes/10))


def get_env_action(state, state_action_value, eps):
    action = None
    if np.random.random() < eps:
        action = random_action()
    else:
        max_val = -1*10**10
        for act in action_set:
            val = get_q_value(state, act, state_action_value)
            if val > max_val:
                max_val = val
                action = act

    return action


def random_action():
    action_index = np.random.choice(len(action_set))
    return action_set[action_index]


def q_update(prev_state, next_state, prev_action, state_action_value, alpha, reward, discount, init=0):
    learned_value = reward + discount * get_action_max(next_state, state_action_value)
    old_value = get_q_value(prev_state, prev_action, state_action_value)
    return old_value + alpha * (learned_value - old_value)


def get_action_max(state, state_action_value, init=0):
    value = init
    if state in state_action_value:
        max_value = -1*10**10
        for action in action_set:
            value = init
            if action in state_action_value[str(state)]:
                value =  state_action_value[str(state)][(action)]
            if value > max_value:
                max_value = value
                action = action

    return value


def get_q_value(state, action, state_action_value, init=0):
    value = init
    if state in state_action_value:
        if action in state_action_value[str(state)]:
            value = state_action_value[str(state)][str(action)]

    return value


def get_state(observation, road_means):
    road_matrix = pre. cropped_grayscale_car_road(observation)
    dashboard_values = pre.get_dashboard_values(observation)
    steering = dashboard_values[0]
    speed = dashboard_values[1]

    speed_state = get_speed_state(speed)
    steering_state = get_steering_state(steering)
    road_state = get_road_state(road_matrix, road_means)

    state = (speed_state, steering_state, road_state)
    return state

def get_speed_state(speed):
    speed_state = 0
    if speed > 0.4:
        speed_state = 1
    return speed_state


def get_steering_state(steering):
    steering_state = 0
    if steering < 0.3:
        steering_state = -1
    elif steering > 0.7:
        steering_state = 1
    return steering_state


def get_road_state(road_matrix, road_means):
    min_dif = road_matrix.size**2
    road_state = -1

    for i_road_mean in range(len(road_means)):
        dif = np.sum(abs(road_matrix - road_means[i_road_mean]))
        if dif < min_dif:
            min_dif = dif
            road_state = i_road_mean
    return road_state


def show_model(model):
    cv2.imshow('model', model)
    cv2.waitKey(1)

    matrix_scaled = model/255
    road_vector = get_main_vector(matrix_scaled)

    print("road_vector", road_vector)
    dim = 40
    img = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            x = i - math.floor(dim/2)
            y = math.floor(dim/2) - j
            v = np.array([x,y])

            if np.sum(v) < 0:
                v *= -1

            guess=road_vector*vector_length(v)

            error_vector = v-guess
            error = vector_length(error_vector)/dim
            img[i][j] = error

    print(img)
    cv2.imshow('road', img)
    cv2.waitKey(1)


if __name__ == "__main__":
    main(10)
