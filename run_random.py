import gym
import myplot
import numpy

def preproc(obs):
    zero = numpy.array([0,0,0])
    other = numpy.array([102,204,102])
    other2 = numpy.array([102,229,102])


    new_list = []
    for col in obs:
      for i in range(len(col)):
        if numpy.array_equal(col[i], zero):
          new_list.append(0)
        elif numpy.array_equal(col[i], other):
            new_list.append(1)
        elif numpy.array_equal(col[i], other2):
            new_list.append(2)
        else:
            print(col[i])
            new_list.append(-1)

    b = numpy.array(new_list).reshape(96,96)
    return b

render = False
n_episodes=1
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

        if(t % 10 == 0):
            preproc(observation)
            # for col in observation:
            #     u = numpy.unique(col, axis=0)
            #     if len(unique_pixels) == 0:
            #         unique_pixels = u
            #     else:
            #          unique_pixel = numpy.concatenate((unique_pixels, u), axis=0)
            #unique_pixels = numpy.unique(unique_pixels,axis=0)
        if (t == 500):
            print(unique_pixels)
            break
        if done or t==999:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break;


#print (rewards)
myplot.plotRewards("Random", rewards, 1)
