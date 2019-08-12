import numpy as np
from driving.car import *
from driving.roadmap import *
from random import random

class QuadraticDistanceAngleReward:
  def __init__(self, map, distance_penalty=1.0, angle_penalty=1.0, action_penalty=1.0):
    self.map = map
    self.distance_penalty = distance_penalty
    self.angle_penalty = angle_penalty
    self.action_penalty = action_penalty

  def reward(self, s, a):
    d, delta = self.map.distance_angle(s[0], s[1], s[2])
    rda = self.reward_distance_angle(d, delta)
    ra = self.reward_action(a)
    return rda + ra

  def reward_distance_angle(self, d, a):
    return -self.distance_penalty*d**2 - self.angle_penalty*a**2

  def reward_action(self, a):
    return -self.action_penalty*a**2

class DrivingEnv:
  def __init__(self,
               map=make_oval(),
               car=DubinsCarModel(),
               dt=0.1,
               reward=QuadraticDistanceAngleReward(make_oval()).reward,
               init_state=np.array((1.0, 0.5, 0.0))):

    self.map = map
    self.car = car
    self.dt = dt
    self.reward = reward
    self.state = init_state

  def step(self, a):
    r = self.reward(self.state, a)
    self.state = self.car.dynamics(self.state, a, self.dt)
    return self.state, r, False, None

  def reset(self):
    x, y = self.map.sample()
    self.state = np.array((x, y, 2*pi*random()))
    return self.state
