import cv2
import numpy as np
from time import sleep
from pyglet.window import key
from matplotlib import pyplot as plt

from sarsa.model import Model
from base.preprocessing import compute_state, vector_size as input_size
from default_action_set import default_action_set

from myplot import plotRewards


from human_env import CarRacing

ACTION = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  ACTION[0] = -1.0
    if k == key.RIGHT: ACTION[0] = +1.0
    if k == key.UP:    ACTION[1] = +1.0
    if k == key.DOWN:  ACTION[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and ACTION[0] == -1.0: ACTION[0] = 0
    if k == key.RIGHT and ACTION[0] == +1.0: ACTION[0] = 0
    if k == key.UP:    ACTION[1] = 0
    if k == key.DOWN:  ACTION[2] = 0


def main():
    env = CarRacing(turn_sharpness=0.05)
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    # env.viewer.window.on_key_press = key_press
    # env.viewer.window.on_key_release = key_release

    model = Model(env, 'sarsa', input_size, default_action_set)

    rewards = []

    for i in range(10):
        reward = play_one(env, model, record_video)
        rewards.append(reward)

    plotRewards("Trained SARSA easy", rewards, radius=2)
    env.close()


def play_one(env, model, record_video):
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False

    while True:
        # if steps == 1:
        #     sleep(2)
        #     break

        state = compute_state(obs)
        action = model.get_trained_action(state)

        obs, r, done, info = env.step(action)


        #
        # if steps==500:
        #     plt.imshow(road_matrix)
        #     plt.show()

        if steps > 1500:
            print("This episode is stuck")
            break
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in ACTION]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.savefig("test.jpeg")
        steps += 1
        if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
            env.render()
        if done or restart: break


if __name__=="__main__":
    main()