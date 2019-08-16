import numpy as np
from driving.car import *
from driving.roadmap import *
from random import random
from matplotlib import pyplot as plt
import gym
from gym.spaces import Box, Discrete


# all angles in this file are in DEGREES unless otherwise noted

class LinearDistanceAngleReward:
  def __init__(self, map, distance_penalty=0.3, angle_penalty=0.0005, action_penalty=0.001):
    self.map = map
    self.distance_penalty = distance_penalty
    self.angle_penalty = angle_penalty
    self.action_penalty = action_penalty

  def reward(self, x,y,theta, a_deg):
    d, delta_deg = self.map.distance_angle_deg(x,y,theta)
    rda = self.reward_distance_angle(d, delta_deg)
    ra = self.reward_action(a_deg)

    r = rda + ra

    # Set the Reward to 0 if it's within 0.1 of the road
    if abs(d) <= 0.1:
      r = 1 + ra

    if abs(d) >= 2.0:
      r -= 20
    return r

  def reward_distance_angle(self, d, delta_deg):
    return -self.distance_penalty*abs(d) - self.angle_penalty*abs(delta_deg)

  def reward_action(self, a_deg):
    return -self.action_penalty*abs(a_deg)


class BadReward:
  def __init__(self, map, distance_penalty=0.3, angle_penalty=0.0005, action_penalty=0.001):
    self.map = map
    self.distance_penalty = distance_penalty
    self.angle_penalty = angle_penalty
    self.action_penalty = action_penalty

  def reward(self, x,y,theta, a_deg):
    d, delta_deg = self.map.distance_angle_deg(x,y,theta)
    rda = self.reward_distance_angle(d, delta_deg)
    ra = self.reward_action(a_deg)

    r = rda + ra

    if abs(d) >= 2.0:
      r -= 20
    return r

  def reward_distance_angle(self, d, delta_deg):
    return -self.distance_penalty*abs(d) - self.angle_penalty*abs(delta_deg)

  def reward_action(self, a_deg):
    return -self.action_penalty*abs(a_deg)

class DrivingEnv(gym.Env):
  """State: [x, y, theta_deg]"""
  def __init__(self,
               map=make_oval(),
               car=DubinsCarModel(),
               dt=0.5,
               reward=BadReward(make_oval()).reward,
               init_state=np.array((1.0, 0.5, 0.0)),
               actions=(-35.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 35.0)):

    self.map = map
    self.car = car
    self.dt = dt
    self.reward = reward
    self.state = init_state
    self.actions = actions
    self.t = 0

  def step(self,action_choice):
    a_deg = self.actions[action_choice]
    return self.step_a_deg(a_deg)

  def step_a_deg(self, a_deg):
    s = self.state
    r = self.reward(s[0], s[1], s[2], a_deg) 
    old_state = self.state
    state_rad = self.state*(1.0, 1.0, pi/180)
    state_rad = self.car.dynamics(state_rad, a_deg*pi/180, self.dt)
    self.state = state_rad*(1.0, 1.0, 180/pi)

    d, ang = self.map.distance_angle_deg(self.state[0], self.state[1], self.state[2])
    if abs(d) >= 2.0:
      done = True
    else:
      done=False

    self.t += 1
    if self.t > 100:
      done = True

    return self.state, r, done, dict()

  def reset(self):
    self.t = 0
    x, y = self.map.sample()
    self.state = np.array((x, y, 360*random()))
    return self.state

  @property
  def observation_space(self):
    return Box(low=-90.0, high=90.0, shape=(3,))

  
  @property
  def action_space(self):
    return Discrete(len(self.actions)) # Change for the Number of the Actions

  # Helper function to get the distance and angle, to help writing reward functions
  def getDistanceAngle(self, x, y, theta):
    return self.map.distance_angle_deg(x, y, theta)

  def distance_angle(self, x, y, theta):
    return self.map.distance_angle_deg(x, y, theta)

def sim(env, policy, n_steps=100):
  s = env.reset()
  history = []
  for i in range(n_steps):
      a = policy(s)
      sp, r, done, info = env.step_a_deg(a)
      history.append((s, a, r, sp))
      s = sp
      if done:
          break
  return history

# def plot_sim(env, policy, n_steps=100):
#     history = sim(env, policy, n_steps)
#     # xs = range(len(history))
#     xs = [step[0][0] for step in history]
#     ys = [step[0][1] for step in history]
#     reward = sum([step[2] for step in history])
#     print(f'reward: {reward}')
#     plt.plot(xs, ys)
#     return history
