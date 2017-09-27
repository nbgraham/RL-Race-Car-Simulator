import gym
import numpy as np

import myplot
import preprocessing as pre

render = True # Does't work if false, observations are wrong
n_episodes = 1
action_time_steps = 5
batch_size = 10

n_hidden = 500
n_actions = 3
dim = 801 # size of list returned from preprocessing

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
                obs = pre.focus_car(observation)
                print(obs.shape)
                action, hidden_layer = forward(obs)

                reward_per_time_step = interval_reward/action_time_steps
                err = error(reward_per_time_step, action)

                l0_list.append(obs)
                l1_list.append(hidden_layer)
                l2_list.append(action)
                error_list.append(err)

                #Reset
                interval_reward = 0
                if t % (batch_size * action_time_steps) == 0:
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

                action[0] = 2*action[0] - 1 # scale steering from [0,1] to [-1,1]


            observation, reward, done, info = env.step(action) # observation is 96x96x3

            interval_reward += reward
            sum_reward += reward
            if t%10==0:
                cropped = observation[:82, 7:89]
                myplot.show_rgb(cropped)

            if (t % 100 == 0):
                print(t)
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
    badness = 1.0/(avg_reward + 1) - 1.0/9.0 # [0,1]
    random_bit = np.random.choice([1,-1])
    dev = np.random.random()*badness/2.0
    change = random_bit*dev
    new_action = action + change
    new_action = new_action % 1.0
    return new_action - action

if __name__ == "__main__":
    main()
