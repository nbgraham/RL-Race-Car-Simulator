import gym
import numpy as np
import json
import cv2
import math

import myplot
import preprocessing as pre


render = True # Does't work if false, observations are wrong

n_episodes = 1
max_time_steps = 2000
action_time_steps = 5
batch_size = 10

target_reward_per_frame = 1.5

def main(k):
    f = open('road_means_' + str(k) + '.npy','rb')
    road_means = np.load(f)
    train(road_means)


def train(road_means):
    env = gym.make('CarRacing-v0')
    action = [0,0.1,0]

    rewards=[]
    for i_episode in range(n_episodes):
        observation = env.reset()
        sum_reward = 0
        interval_reward = 0

        for t in range(max_time_steps):
            if render: env.render()

            if t> 0 and (t % action_time_steps == 0):
                action = env.action_space.sample() # [steering, gas, brake]

                road_matrix = pre. cropped_grayscale_car_road(observation)
                dashboard_values = pre.get_dashboard_values(observation)
                steering = dashboard_values[0]
                speed = dashboard_values[1]

                speed_state = 0
                if speed > 0.4:
                    speed_state = 1

                steering_state = 0
                if steering < 0.3:
                    steering_state = -1
                elif steering > 0.7:
                    steering_state = 1

                min_dif = 10**10
                road_state = -1

                for i_road_mean in range(len(road_means)):
                    dif = np.sum(abs(road_matrix - road_means[i_road_mean]))
                    if dif < min_dif:
                        min_dif = dif
                        road_state = i_road_mean

                cv2.imshow('road state', road_means[road_state])
                cv2.waitKey(1)

                print("Road: {} Steering: {} Speed: {}".format(road_state, steering_state, speed_state))

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

    f = open('rewards/rewards','w')
    json.dump(rewards, f)
    f.close()

    myplot.plotRewards("Random", rewards, int(n_episodes/10))


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
