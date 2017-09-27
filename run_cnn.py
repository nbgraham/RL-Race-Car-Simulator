import gym
import myplot
import numpy as np

render = True # Does't work if false, observations are wrong
n_episodes = 1
action_time_steps = 5
batch_size = 10

n_hidden = 500
n_actions = 3
dim = 96*96

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
                obs = preproc(observation)
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
            new_list.append(-1)
        elif np.array_equal(col[i], light_green):
            new_list.append(-1.1)
        elif np.array_equal(col[i], grey1):
            new_list.append(2)
        elif np.array_equal(col[i], grey2):
            new_list.append(2.1)
        elif np.array_equal(col[i], grey3):
            new_list.append(1.9)
        else:
            print("Unexpected pixel: ", col[i])
            new_list.append(3)

    b = np.array(new_list)
    return b

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
