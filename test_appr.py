import numpy as np

from apprentice.model import Model

from base.preprocessing import compute_state, vector_size as input_size
from apprentice.action_set import default_action_set

from human_env import CarRacing

def main():
    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('monitors/appr/video-test', force=True)

    model = Model(env, 'appr', input_size, default_action_set)

    for i in range(10):
        play_one(env, model, record_video)

    env.close()


def play_one(env, model, record_video):
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    restart = False

    prev_state = compute_state(observation)
    print(default_action_set)

    while True:
        probs = model.predict(prev_state)
        print(probs)
        action_index = np.argmax(probs)
        action = default_action_set[action_index]

        observation, r, done, info = env.step(action)

        print(action)

        prev_state = compute_state(observation)

        total_reward += r
        if steps % 200 == 0 or done:
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1

        env.render()
        if done or restart: break


if __name__=="__main__":
    main()