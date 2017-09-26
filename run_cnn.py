import gym
import myplot
import numpy as np

render = False
n_episodes = 1
action_time_steps = 5

n_hidden = 500
n_actions = 3
dim = 96*96

min_reward_per_frame = -0.1
max_reward_per_frame = 10

model = {}
model['W1'] = np.random.randn(n_hidden, dim)
model['W2'] = np.random.randn(n_actions, n_hidden)

def main():
    env = gym.make('CarRacing-v0')

    rewards=[]
    for i_episode in range(n_episodes):
        observation = env.reset()
        sum_reward = 0
        interval_reward = 0

        action = [0,0,0] # [steering, gas, brake]
        for t in range(1000):
            if render: env.render()

            if (t % action_time_steps == 0):
                o = preproc(observation)
                action = cnn(o)
                interval_reward = 0

            observation, reward, done, info = env.step(action) # observation is 96x96x3

            interval_reward += reward
            sum_reward += reward

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


def preproc(obs):
    black = np.array([0,0,0])
    green = np.array([102,204,102])
    light_green = np.array([102,229,102])
    grey1 = np.array([107,107,107])
    grey2 = np.array([105,105,105])

    new_list = []
    for col in obs:
      for i in range(len(col)):
        if np.array_equal(col[i], black):
            new_list.append(1)
        elif np.array_equal(col[i], green):
            new_list.append(2)
        elif np.array_equal(col[i], light_green):
            new_list.append(2.1)
        elif np.array_equal(col[i], grey1):
            new_list.append(3)
        elif np.array_equal(col[i], grey2):
            new_list.append(3)
        else:
            print(col[i])
            new_list.append(0)

    b = np.array(new_list)
    return b

def cnn(obs):
    one = np.dot(model['W1'], obs)
    one[one < 0] = 0 #ReLu non linearity
    two = np.dot(model['W2'], one)

    two[0] = posNegNorm(two[0]) #steering
    two[1] = posNorm(two[1]) #gas
    two[2] = posNorm(two[2]) #brake
    return two

def posNorm(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid squashing

def posNegNorm(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0 # sigmoid squashing

if __name__ == "__main__":
    main()