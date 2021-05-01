# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:41:51 2021

@author: mclea
"""

import gym

env = gym.make("CartPole-v1")
observation = env.reset()

print(observation)
print(env.action_space)

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(env.action_space.sample())
    env.render()

env.close()