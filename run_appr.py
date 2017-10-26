import cv2
import numpy as np
from time import sleep
from pyglet.window import key
from matplotlib import pyplot as plt

from apprentice.model import Model

from base.preprocessing import compute_state, vector_size as input_size
from apprentice.action_set import default_action_set

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
    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('monitors/appr/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    model = Model(env, 'appr', input_size, default_action_set)

    for i in range(10):
        play_one(env, model, record_video)
        model.save()

    env.close()


def play_one(env, model, record_video):
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False

    prev_state = compute_state(observation)
    action_index = 0
    print(default_action_set)

    while True:
        model.encourage_action(prev_state, action_index)
        observation, r, done, info = env.step(ACTION)

        for i in range(len(default_action_set)):
            act = default_action_set[i]
            if (ACTION == act).all():
                action_index = i

        prev_state = compute_state(observation)

        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in ACTION]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1

        env.render()
        if done or restart: break


if __name__=="__main__":
    main()