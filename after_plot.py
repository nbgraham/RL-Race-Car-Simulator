import json
import myplot

f = open('rewards')
rewards = json.load(f)

myplot.plotRewards("Simple CNN",rewards,10)
