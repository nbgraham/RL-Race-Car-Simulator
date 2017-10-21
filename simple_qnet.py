import tensorflow as tf
import numpy as np
from collections import deque
import random

#network parameters
num_input = 10*10+7+4 # size of list returned from preprocessing
num_hidden = 512

#Constants
LEARNING_RATE = 0.001
BATCH_SIZE = 100


class SimpleQNet:
    def __init__(self,action_set):
        # tf graph input
        self.stateInput = tf.placeholder("float", [None, num_input])

        # layers and weight bias
        self.weights = {
            'hidden': tf.Variable(tf.random_normal([num_input, num_hidden])),
            'output': tf.Variable(tf.random_normal([num_hidden, len(action_set)]))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random_normal([num_hidden])),
            'output': tf.Variable(tf.random_normal([len(action_set)]))
        }

        # establish forward feed part of network to choose actions
        hidden = tf.nn.relu(tf.add(tf.matmul(self.stateInput, self.weights['hidden']), self.biases['hidden']))
        self.qvals = tf.add(tf.matmul(hidden, self.weights['output']), self.biases['output'])

    def properties(self):
        return (self.stateInput,self.weights,self.biases,self.qvals)

class SQN:
    def __init__(self,action_set):
        self.replayMemory = deque()
        self.actions = action_set

        self.currentQNet = SimpleQNet(self.actions)
        self.targetQNet = SimpleQNet(self.actions)

        self.actionInput = tf.placeholder("float",[None,len(action_set)])
        self.yInput = tf.placeholder("float",[None])
        self.Q_action = tf.reduce_sum(tf.matmul(self.currentQNet.qvals,self.actionInput),reduction_indices=1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.yInput,self.Q_action))
        self.trainStep = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def copyCurrentToTargetNet(self):
        targetProperties = self.targetQNet.properties()
        currentProperties = self.currentQNet.properties()
        properties = zip(targetProperties,currentProperties)
        return [tvar.assign(cvar) for tvar,cvar in properties]

    def selectAction(self,currentState,epsilon):
        if random.random() < epsilon:
            action_index = random.randrange(0,len(self.actions))
        else:
            qout = self.currentQNet.q_value.eval(feed_dict={self.stateInput:[currentState]})
            action_index = np.argmax(qout)
        return self.actions[action_index]

    def storeExperience(self, state, action, reward, stateprime, terminalstate):
        self.replayMemory.append((state,action,reward,stateprime,terminalstate))

    def sampleExperiences(self):
        if len(self.replayMemory) < BATCH_SIZE:
            return list(self.replayMemory)
        return random.sample(self.replayMemory, BATCH_SIZE)
