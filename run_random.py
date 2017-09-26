import gym
import myplot
import numpy as np

render = False
n_episodes=1

n_hidden = 500
n_actions = 3
dim = 96*96

model = {}
model['W1'] = np.random.rand(n_hidden, dim)
model['W2'] = np.random.rand(n_hidden, n_actions)

def main():
    env = gym.make('CarRacing-v0')

    rewards=[]
    for i_episode in range(n_episodes):
        observation = env.reset()
        sum_reward = 0
        unique_pixels = []

        for t in range(1000):
            if render: env.render()
            action = env.action_space.sample() # [steering, gas, brake]
            observation, reward, done, info = env.step(action) # observation is 96x96x3
            sum_reward += reward

            if (t==233):
                o = preproc(observation)
                t = cnn(o)
                print(t)
                break
            if(t % 5 == 0):
                preproc(observation)
            if (t % 100 == 0):
                print(t)
            if done or t==999:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break;

    #print (rewards)
    myplot.plotRewards("Random", rewards, 1)


def preproc(obs):
    zero = np.array([0,0,0])
    other = np.array([102,204,102])
    other2 = np.array([102,229,102])

    new_list = []
    for col in obs:
      for i in range(len(col)):
        if np.array_equal(col[i], zero):
          new_list.append(0)
        elif np.array_equal(col[i], other):
            new_list.append(1)
        elif np.array_equal(col[i], other2):
            new_list.append(2)
        else:
            print(col[i])
            new_list.append(-1)

    b = np.array(new_list)
    return b

def cnn(obs):
    one = np.dot(model['W1'], obs)
    one[one < 0] = 0 #ReLu non linearity
    two = np.dot(model['W2'], one)
    return two

if __name__ == "__main__":
    main()
