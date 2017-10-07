import gym
import numpy as np
import json
import random

import myplot
import preprocessing as pre


n_episodes = 500
max_time_steps = 2000
action_time_steps = 3
batch_size = 10
target_reward_per_frame = 1.5
min_reward_per_frame = -0.1
easiness = 5

epochs = 1000
gamma = 0.9
epsilon = 0.1

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
#brake
[0,0,0.5],
[0,0,0.8]
])

n_hidden = 512
n_outputs = len(action_set)
input_dim = 10*10+7+4 # size of list returned from preprocessing


err_a = (1-np.exp(-1))/(target_reward_per_frame-min_reward_per_frame)
err_b = np.exp(-1)-err_a


# initialize model [-1,1] with mean 0
np.random.seed(35)
model = {}
model['W1'] = 2 * np.random.random((input_dim, n_hidden)) - 1
model['W2'] = 2 * np.random.random((n_hidden, n_outputs)) - 1

def main():
    env = gym.make('CarRacing-v0')

    rewards = []
    for i_episode in range(n_episodes):

        observation = env.reset()
        sum_reward = 0
        interval_reward = 0

        input_layer_list = []
        hidden_layer_list = []
        output_layer_list = []
        error_list = []

        # default action
        action_selector = np.zeros((1,len(action_set))) #fill array with 0s
        action_selector[:,0] = 1 # default to nothing action
        action = np.dot(action_selector, action_set).ravel()  # [steering, gas, brake]

        for t in range(max_time_steps):
            env.render()

            if t > 0 and (t % action_time_steps == 0):
                state = pre.compute_state(observation)

                action_probability, hidden_layer = feedfwd(state)

                # calc reward per timestep
                reward_per_time_step = interval_reward/action_time_steps
                # use prev action selection to calculate error
                err = error(reward_per_time_step, action_selector, action_probability)

                # get new action selection with random probability epsilon
                action_selector, qval = get_env_action(action_probability,epsilon)
                action = np.dot(action_selector, action_set).ravel()

                input_layer_list.append(state)
                hidden_layer_list.append(hidden_layer)
                output_layer_list.append(action_probability)
                error_list.append(err)

                interval_reward = 0

                if t % (batch_size * action_time_steps) == 0:

                    input_layer_array = np.vstack(input_layer_list)
                    hidden_layer_array = np.vstack(hidden_layer_list)
                    output_layer_array = np.vstack(output_layer_list)
                    error_array = np.vstack(error_list)

                    if t %250 == 0:
                        print("  Time step:", t)
                        print("  Mean error")
                        print("  ", np.mean(error_array,axis=0))

                    output_layer_delta = error_array * sigmoidDeriv(output_layer_array)
                    hidden_layer_error = output_layer_delta.dot(model['W2'].T)
                    hidden_layer_delta = hidden_layer_error * sigmoidDeriv(hidden_layer_array)

                    model['W2'] += hidden_layer_array.T.dot(output_layer_delta)
                    model['W1'] += input_layer_array.T.dot(hidden_layer_delta)

                    input_layer_list = []
                    hidden_layer_list = []
                    output_layer_list = []
                    error_list = []

            observation, reward, done, info = env.step(action)

            interval_reward += reward
            sum_reward += reward

            if done or t==max_time_steps-1:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break

    f = open('rewards/rewards_fake','w')
    json.dump(rewards, f)
    f.close()

    myplot.plotRewards("qval", rewards, int(n_episodes/10))


def get_env_action(nn_output,eps):
    qval = nn_output
    if np.random.random() < eps:
        action_selector = np.zeros((1,len(action_set)))
        action_selector[:,random.randint(0,len(action_set)-1)] = 1
    else:
        total = np.sum(nn_output)
        prob = nn_output/total
        action_index = np.random.choice(len(action_set), p=prob)
        action_selector = np.zeros((1,len(action_set)))
        action_selector[:,action_index] = 1
    return action_selector, qval

# sigmoid squashing (value between 0 and 1)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#derivative of sigmoid
def sigmoidDeriv(x):
    #assuming x is a result sigmoid(y)
    return x*(1-x)

#pass state through neural net to get action probabilities
def feedfwd(inputs):
    #hidden = relu(np.dot(inputs,model['W1']))
    hidden = sigmoid(np.dot(inputs,model['W1']))
    aprob = sigmoid(np.dot(hidden,model['W2']))
    return aprob, hidden #return probability of taking action 2 and hidden state

#set anything below 0 to 0
def relu(vector):
    vector[vector<0] = 0
    return vector

#avg-reward is reward per timestep, action selector is array of 0's with one 1, nn_prob is output layer weights
def error(avg_reward, action_selector, nn_prob):
    #take reward per timestep (add 1 to make positive?) subtract minimum reward per frame (-0.1)
    scaled_x = (avg_reward+1-min_reward_per_frame)
    #(err_a * scaled_x + err_b)^easiness
    badness = np.log(err_a*scaled_x + err_b)**easiness # [-1.0], [bad.good]

    selector_delta = np.copy(action_selector)
    selector_delta[selector_delta == 0] = -1
    selector_delta *= badness

    selector_target = np.clip(nn_prob + selector_delta,a_min=0,a_max=1)

    error = selector_target - nn_prob

    return error

if __name__ == "__main__":
    main()
