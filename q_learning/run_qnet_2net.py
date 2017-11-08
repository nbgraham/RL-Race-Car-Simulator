import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
import tensorflow as tf
from gym import wrappers

''' 
fourth experiment:
simple q-learning agent

has a current and target network 
the target is used to generate actions with current being updated based on those actions, copied over every 100 frames

uses exponential epsilon decay
'''

env = gym.make('CarRacing-v0')
env = wrappers.Monitor(env, 'monitor-folder', force=True)

num_episodes = 1
max_time_steps = 1500
update_time_steps = 100

#learning parameters
learning_rate = 0.01
gamma = 0.99
epsilon = 1 #starting at 1 so random all of the time (lowering as episodes increase)
min_epsilon = 0.1


action_set = np.array([
#steering (left,right)
[-1.0,0,0],
[-0.75,0,0],
[-0.5,0,0],
[-0.25,0,0],
[0,0,0],
[0.25,0,0],
[0.5,0,0],
[0.75,0,0],
[1.0,0,0],
#gas
[0,0.33,0],
#brake
[0,0,0.5]
])

#network parameters
num_input = 10*10+7+4 # size of list returned from preprocessing
num_hidden = 512
num_output = len(action_set)

class net():
    def __init__(self):
        #tf graph input
        self.x = tf.placeholder("float",[None,num_input])

        #layers and weight bias
        self.weights = {
            'hidden':tf.Variable(tf.random_normal([num_input,num_hidden])),
            'output': tf.Variable(tf.random_normal([num_hidden, num_output]))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random_normal([num_hidden])),
            'output': tf.Variable(tf.random_normal([num_output]))
        }

        #establish forward feed part of network to choose actions
        self.hidden = tf.nn.relu(tf.add(tf.matmul(self.x,self.weights['hidden']),self.biases['hidden']))
        self.qvals = tf.add(tf.matmul(self.hidden,self.weights['output']),self.biases['output'])

        #update model based on loss
        self.next_qvals = tf.placeholder("float",shape=[1,num_output])
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_qvals,self.qvals))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_model = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()


    def properties(self):
        return self.weights['hidden'],self.weights['output'],self.biases['hidden'],self.biases['output']

def copyCurrentToTargetNet(current,target):
    targetProperties = target.properties()
    currentProperties = current.properties()
    properties = zip(targetProperties,currentProperties)
    return [tvar.assign(cvar) for tvar,cvar in properties]

def get_env_action(nn_output, eps):
    if np.random.random() < eps:
        action_index = random.randint(0, len(action_set)-1)
    else:
        action_index = np.argmax(nn_output)
    return action_set[action_index]

#current and target neural nets
cnet = net()
tnet = net()

with tf.Session() as sess:
    sess.run(cnet.init)
    sess.run(tnet.init)

    totalrewards = np.empty(num_episodes)
    totallosses = np.empty(num_episodes)
    for episode in range(num_episodes):
        # decrease epsilon every episode
        if (epsilon >= min_epsilon):
            epsilon -= 0.001

        #reset for each episode
        observation = env.reset()
        done = False
        totalreward = 0
        timesteps = 0
        totalloss = 0

        #run episode until done (or until max time steps)
        while not done:
            env.render()

            state = pre.compute_state(observation)

            #select action using current qnet
            q_values = sess.run(cnet.qvals, feed_dict={cnet.x: state.reshape(-1,num_input)})[0]

            action = get_env_action(q_values,epsilon)

            observation, reward, done, info = env.step(action)
            prev_state = state
            state = pre.compute_state(observation)

            #determine target action using target qnet
            q_prime = sess.run(tnet.qvals,feed_dict={tnet.x:state.reshape(-1,num_input)})[0]
            if not done:
                q_target = reward + gamma * np.max(q_prime)
            else:#set terminal states to just the reward
                q_target = reward
            target = q_values[:]
            target[np.argmax(q_values)] = q_target

            #update current qnet
            _,cost = sess.run([cnet.update_model,cnet.loss],feed_dict={cnet.x:prev_state.reshape(-1,num_input),cnet.next_qvals:target.reshape(-1,num_output)})


            totalloss += cost
            totalreward += reward
            timesteps += 1

            #set target qnet to current qnet
            if timesteps % update_time_steps == 0:
                sess.run(copyCurrentToTargetNet(cnet,tnet))
            if timesteps > max_time_steps:
                print("too many time steps, breaking")
                break

        totallosses[episode] = totalloss
        totalrewards[episode] = totalreward
        if episode % 1 == 0:
            print("episode:", episode, "timesteps:", timesteps, "total reward:", totalreward, "eps:", epsilon, "avg reward (last 100):",totalrewards[max(0,episode-100):(episode+1)].mean())
        print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("target network", totalrewards, int(num_episodes/10))
    mp.show()
