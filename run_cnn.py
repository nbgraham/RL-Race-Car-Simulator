import gym
import numpy as np

import myplot
import preprocessing as pre

render = True # Does't work if false, observations are wrong
n_episodes = 1
action_time_steps = 5
batch_size = 10

target_reward_per_frame = 1

n_hidden = 5
n_actions = 3
dim = 9 # size of list returned from preprocessing

np.random.seed(35)
model = {}
# initialize [-1,1] with mean 0
model['W1'] = 2 * np.random.random((dim, n_hidden)) - 1
model['W2'] = 2 * np.random.random((n_hidden, n_actions)) - 1

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
        error_list = []

        action = [0,0,0] # [steering, gas, brake]
        for t in range(1000):
            if render: env.render()

            if (t % action_time_steps == 0):
                obs = pre.coarse(observation).ravel()/255
                l2, hidden_layer = forward(obs)

                reward_per_time_step = interval_reward/action_time_steps
                err = error(reward_per_time_step, l2)

                l0_list.append(obs)
                l1_list.append(hidden_layer)
                l2_list.append(l2)
                error_list.append(err)

                if t % (batch_size * action_time_steps) == 0:
                    print("Time step:", t)
                    print("Action: ", action)
                    print("Error:",  err)

                    l0_array = np.vstack(l0_list)
                    l1_array = np.vstack(l1_list)
                    l2_array = np.vstack(l2_list)
                    error_array = np.vstack(error_list)

                    l2_delta = error_array * sigmoidDeriv(l2_array)
                    l1_error = l2_delta.dot(model['W2'].T)
                    l1_delta = l1_error * sigmoidDeriv(l1_array)

                    model['W2'] += l1_array.T.dot(l2_delta)
                    model['W1'] += l0_array.T.dot(l1_delta)

                    l0_list = []
                    l1_list = []
                    l2_list = []
                    error_list = []

                interval_reward = 0

                #copy l2 so we can modify it
                action = np.empty_like(l2)
                action[:] = l2

                action[0] = 2*action[0] - 1 # scale steering from [0,1] to [-1,1]


            observation, reward, done, info = env.step(action) # observation is 96x96x3

            interval_reward += reward
            sum_reward += reward

            if done or t==999:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break;

    #print (rewards)
    #myplot.plotRewards("Random", rewards, 1)

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
