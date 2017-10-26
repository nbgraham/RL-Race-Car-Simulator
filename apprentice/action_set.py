default_action_set = []

for steering in [-1.0, 0.0, 1.0]:
    for gas in [0.0,1.0]:
        for brake in [0.0, 0.8]:
            default_action_set.append([steering, gas, brake])
