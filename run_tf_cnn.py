import gym
import numpy as np
import random
import myplot as mp
import preprocessing as pre
import tensorflow as tf
from gym import wrappers

''' 
fifth experiment:
cnn q-learning agent
'''

env = gym.make('CarRacing-v0')
env = wrappers.Monitor(env, 'monitor-folder', force=True)
#repeats an action for 3 frames with env.step(action)
action_time_steps = 3
frame_skip = wrappers.SkipWrapper(action_time_steps)
env = frame_skip(env)

num_episodes = 1000
max_time_steps = 1500
batch_size = 10
update_time_steps = 10

#learning parameters
learning_rate = 0.01
gamma = 0.99
epsilon = 0.8 #lowering as episodes increase
min_epsilon = 0.05

action_set = np.array([
#steering (left,right)
[-1.0,0,0],
[-0.5,0,0],
[0,0,0],
[0.5,0,0],
[1.0,0,0],
#gas
[0,0.33,0],
#brake
[0,0,0.5]
])

#network parameters
num_channels = 4
image_size = 84
num_output = len(action_set)
random_seed = 35

def weight_variable(shape,stdev=0.1):
    initial = tf.truncated_normal(shape, stddev=stdev,seed=random_seed)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)

class cnn:
    def __init__(self):
        #weights and biases reassigned during training

        self.conv1_w  = weight_variable([8,8,num_channels,32])
        self.conv1_b = bias_variable([32])

        self.conv2_w = weight_variable([4, 4, 32, 64])
        self.conv2_b = bias_variable([64])

        self.conv3_w = weight_variable([3, 3, 64, 64])
        self.conv3_b = bias_variable([64])

        self.fc_w = weight_variable([7744,512])
        self.fc_b = bias_variable([512])

        self.num_actions = num_output
        self.out_w = weight_variable([512,num_output])
        self.out_b = bias_variable([num_output])

        self.stateInput = tf.placeholder("float",[None,image_size,image_size,num_channels])

        # hidden layers
        h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.stateInput,self.conv1_w,strides=[1,4,4,1],padding='SAME'),self.conv1_b))

        h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv1,self.conv2_w,strides=[1,2,2,1],padding='SAME'),self.conv2_b))

        h_conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv2,self.conv3_w,strides=[1,1,1,1],padding='SAME'),self.conv3_b))

        relu_shape = h_conv3.get_shape().as_list()
        # print(relu_shape)
        reshape = tf.reshape(h_conv3, [-1,relu_shape[1]*relu_shape[2]*relu_shape[3]])

        #fully connected and output layers
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc_w)+self.fc_b)
        self.qvals = tf.add(tf.matmul(hidden,self.out_w),self.out_b)

        # update model based on loss
        self.next_qvals = tf.placeholder("float",shape=[1,num_output])
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_qvals,self.qvals))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_model = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()


    def properties(self):
        return (self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b, self.conv3_w, self.conv3_b, self.fc_w, self.fc_b,
        self.out_w, self.out_b)

def copyCurrentToTargetNet(current,target):
    targetProperties = target.properties()
    currentProperties = current.properties()
    properties = zip(targetProperties,currentProperties)
    return [tvar.assign(cvar) for tvar,cvar in properties]

# saver = tf.train.Saver()
# model_path = "./model/car.ckpt"

def get_env_action(nn_output, eps):
    if np.random.random() < eps:
        action_index = random.randint(0, len(action_set)-1)
    else:
        action_index = np.argmax(nn_output)
    return action_set[action_index]

#current and target neural nets
cnet = cnn()
tnet = cnn()

with tf.Session() as sess:
    sess.run(cnet.init)
    sess.run(tnet.init)

    totalrewards = np.empty(num_episodes)
    totallosses = np.empty(num_episodes)
    for episode in range(num_episodes):
        #decrease epsilon every episode
        if (epsilon > min_epsilon):
            epsilon -= 0.001

        #reset for each episode
        observation = env.reset()
        state = pre.above_bar_no_resize(observation)
        state = np.stack((state,state,state,state),axis=2).reshape(84,84,4)

        done = False
        totalreward = 0
        timesteps = 0
        totalloss = 0

        #run episode until done (or until max time steps)
        while not done:
            env.render()

            # state = pre.above_bar_no_resize(observation)

            #select action using current qnet
            q_values = sess.run(cnet.qvals, feed_dict={cnet.stateInput: [state]})[0]

            action = get_env_action(q_values,epsilon)

            observation, reward, done, info = env.step(action)

            prev_state = state
            state = pre.above_bar_no_resize(observation)
            state = np.append(state,prev_state,axis=2)[:,:,1:]


            #determine target action using target qnet
            q_prime = sess.run(tnet.qvals,feed_dict={tnet.stateInput:[state]})[0]
            if not done:
                q_target = reward + gamma * np.max(q_prime)
            else:#set terminal states to just the reward
                q_target = reward
            target = q_values[:]
            target[np.argmax(q_values)] = q_target

            #update current qnet
            _,cost = sess.run([cnet.update_model,cnet.loss],feed_dict={cnet.stateInput:[prev_state],cnet.next_qvals:target.reshape(-1,num_output)})


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
        # if episode % batch_size == 0:
        #     save_path = saver.save(sess, model_path)  #need to save model weights maybe?
        #     print("model saved in file: ", save_path,"\n")
        #     # can load later with saver.restore(sess, model_path)
        print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
        print("total steps:", totalrewards.sum())

    mp.plotRewards("convolutional qnet", totalrewards, int(num_episodes/10))
    mp.show()
