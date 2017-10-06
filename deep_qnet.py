import tensorflow as tf
import numpy as np
from collections import deque
import random

# basically this entire class
# https://github.com/noahgolmant/DQN/blob/master/dqn.py

# might try to use some stuff from this?
# https://github.com/asrivat1/DeepLearningVideoGames/blob/master/deep_q_network.py


#Constants
NUM_CHANNELS = 4 #image channels
IMAGE_SIZE = 80 #80*80 pixel images
SEED = 35 #random initialization seed
NUM_ACTIONS = 4 #gas, left, right, brake
BATCH_SIZE = 100
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
GAMMA = 0.99
LEARNING_RATE = 0.001


def weight_variable(shape,stdev=0.1)
    initial = tf.truncated_normal(shape, stddev=stdev,seed=SEED)
    return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)

class DeepQNet:
    def __init__(self,num_actions):
        #weights and biases reassigned during training

        self.conv1_w  = weight_variable([8,8,NUM_CHANNELS,32])
        self.conv1_b = bias_variable([32])

        self.conv2_w = weight_variable([4, 4, 32, 64])
        self.conv2_b = bias_variable([64])

        self.conv3_w = weight_variable([3, 3, 64, 64])
        self.conv3_b = bias_variable([32])

        self.fc_w = weight_variable([6400,512])
        self.fc_b = bias_variable([512])

        self.num_actions = num_actions
        self.out_w = weight_variable([512,num_actions])
        self.out_b = bias_variable([num_actions])

        self.stateInput = tf.placeholder("float",[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])

        # hidden layers
        #info about them (coming soon tm)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.stateInput,self.conv1_w,strides=[1,4,4,1],padding='SAME'),self.conv1_b))

        h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv1,self.conv2_w,strides=[1,2,2,1],padding='SAME'),self.conv2_b))

        h_conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv2,self.conv3_w,strides=[1,1,1,1],padding='SAME'),self.conv3_b))

        relu_shape = h_conv3.get_shape().as_list()
        print(relu_shape)
        reshape = tf.reshape(h_conv3, [-1,relu_shape[1]*relu_shape[2],relu_shape[3]])

        #fully connected and output layers
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc_w)+self.fc_b)
        self.q_value = tf.add(tf.matmul(hidden,self.out_w),self.out_b)

    def properties(self):
        return (self.conv1_w,self.conv1_b,self.conv2_w,self.conv2_b,self.conv3_w,self.conv3_b,self.fc_w,self.fc_b,self.out_w,self.out_b)

class DQN:
    def __init__(self, actions):
        self.replayMemory = deque()
        self.timestep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        self.currentQNet = DeepQNet(len(actions))
        self.targetQNet = DeepQNet(len(actions))

        self.actionInput = tf.placeholder("float",[None,len(actions)])
        self.yInput = tf.placeHolder("float",[None])
        self.Q_action = tf.reduce_sum(tf.mul(self.currentQNet.q_value,self.actionInput),reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.yInput-self.Q_action))
        self.trainStep = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def copyCurrentToTargetNet(self):
        targetProperties = self.targetQNet.properties()
        currentProperties = self.currentQNet.properties()
        properties = zip(targetProperties,currentProperties)
        return [tvar.assign(cvar) for tvar,cvar in properties]

    #maybe need to change (or nah)
    def selectAction(self,currentState):
        action = np.zeros(len(self.actions))
        if random.random() < self.epsilon:
            action_index = random.randrange(0,len(self.actions))
        else:
            qout = self.currentQNet.q_value.eval(feed_dict={self.stateInput:[currentState]})
            action_index = np.argmax(qout)
        action[action_index] = 1.0
        return action

    def storeExperience(self, state, action, reward, stateprime, terminalstate):
        self.replayMemory.append((state,action,reward,stateprime,terminalstate))

    def sampleExperience(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return list(self.replayMemory)
        return random.sample(self.replayMemory, BATCH_SIZE)
