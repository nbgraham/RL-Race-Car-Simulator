import gym
import numpy as np
import json

import myplot
import preprocessing as pre

render = True # Does't work if false, observations are wrong

n_episodes = 50
max_time_steps = 2000
action_time_steps = 5
batch_size = 10
target_reward_per_frame = 1.5

action_set = np.array([
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
[0,0,0.8],
#nothing
[0,0,0]
])

n_hidden = 25
n_outputs = len(action_set)
input_dim = 9*9+7 # size of list returned from preprocessing

# initialize model [-1,1] with mean 0
np.random.seed(35)
model = {}
model['W1'] = 2 * np.random.random((input_dim, n_hidden)) - 1
model['W2'] = 2 * np.random.random((n_hidden, n_outputs)) - 1

def main():
    env = gym.make('CarRacing-v0')

    rewards=[]
    for i_episode in range(n_episodes):
        observation = env.reset()
        sum_reward = 0
        interval_reward = 0

        l0_list = []
        l1_list = []
        l2_list = []
        action_list = []
        error_list = []
        action = [0,0,0] # [steering, gas, brake]

        for t in range(max_time_steps):
            if render: env.render()

            if t> 0 and (t % action_time_steps == 0):
                coarse_road = pre.coarse(observation).ravel()/255
                dashboard_values = pre.get_dashboard_values(observation)
                obs = np.hstack([coarse_road, dashboard_values])

                l2, hidden_layer = forward(obs)

                reward_per_time_step = interval_reward/action_time_steps
                err = error(reward_per_time_step, l2)
                action_selector = get_env_action(l2)
                action = np.dot(action_selector, action_set).ravel()

                l0_list.append(obs)
                l1_list.append(hidden_layer)
                l2_list.append(l2)
                action_list.append(action_selector)
                error_list.append(err)

                interval_reward = 0

                if t % (batch_size * action_time_steps) == 0:

                    l0_array = np.vstack(l0_list)
                    l1_array = np.vstack(l1_list)
                    l2_array = np.vstack(l2_list)
                    error_array = np.vstack(error_list)

                    if t %250 == 0:
                        print("  Time step:", t)
                        print("  Mean error")
                        print("  ", np.mean(error_array,axis=0))

                    l2_delta = error_array * sigmoidDeriv(l2_array)
                    l1_error = l2_delta.dot(model['W2'].T)
                    l1_delta = l1_error * sigmoidDeriv(l1_array)

                    model['W2'] += l1_array.T.dot(l2_delta)
                    model['W1'] += l0_array.T.dot(l1_delta)

                    l0_list = []
                    l1_list = []
                    l2_list = []
                    error_list = []

            observation, reward, done, info = env.step(action) # observation is 96x96x3

            interval_reward += reward
            sum_reward += reward

            if done or t==max_time_steps-1:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break

    f = open('rewards','w')
    json.dump(rewards, f)
    f.close()

    myplot.plotRewards("Random", rewards, int(n_episodes/10))


def get_env_action(nn_output):
    total = np.sum(nn_output)
    prob = nn_output/total
    action_index = np.random.choice(len(action_set), p=prob)
    action_selector = np.zeros((1,len(action_set)))
    action_selector[:,action_index] = 1
    return action_selector


def forward(l0):
    l1 = sigmoid(np.dot(l0, model['W1']))
    l2 = sigmoid(np.dot(l1, model['W2']))

    return l2, l1

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid squashing

def sigmoidDeriv(x):
    #assuming x is a result sigmoid(y)
    return x*(1-x)

def error(avg_reward, action):
    badness = (target_reward_per_frame - avg_reward)/1.1
    random_bit = np.random.choice([1,-1])
    dev = badness/2.0
    change = random_bit*dev
    new_action = action + change
    new_action = new_action % 1.0
    return new_action - action

if __name__ == "__main__":
    main()
