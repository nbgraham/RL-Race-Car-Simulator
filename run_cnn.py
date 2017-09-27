import gym
import myplot
import numpy as np

render = False
n_episodes = 1
action_time_steps = 5

n_hidden = 500
n_actions = 3
dim = 96*96

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
                print(action)
                print(interval_reward)
                reward_per_time_step = interval_reward/action_time_steps
                err = error(reward_per_time_step)
                print(err)
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
    grey3 = np.array([102,102,102])

    #new things (from unexpected pixels)
    #not sure how to append them to the new list
    red = np.array([255,0,0])
    red2 = np.array([204, 0, 0])
    white = np.array([255,255,255])
    lime = np.array([0,255,0])
    grey4 = np.array([228,228,228])
    grey5 = np.array([206,206,206])
    grey6 = np.array([111,111,111])
    grey7 = np.array([204,204,204])
    grey8 = np.array([251,251,251])
    grey9 = np.array([139,139,139])
    black2 = np.array([81, 81, 81])
    black3 = np.array([38, 38, 38])
    black4 = np.array([33, 33, 33])
    black5 = np.array([26, 26, 26])
    black6 = np.array([45,45,45])



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
            new_list.append(2.9)
        elif np.array_equal(col[i], grey2):
            new_list.append(3)
        elif np.array_equal(col[i], grey3):
            new_list.append(3.1)
        else:
            print("Unexpected pixel: ", col[i])
            new_list.append(3)

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

def error(avg_reward):
    return 1.0/(avg_reward + 1) - 1.0/9.0

if __name__ == "__main__":
    main()
