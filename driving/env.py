import numpy as np
from driving.car import *
from driving.roadmap import *
from random import random

# all angles in this file are in DEGREES unless otherwise noted

class QuadraticDistanceAngleReward:
  def __init__(self, map, distance_penalty=1.0, angle_penalty=0.0001, action_penalty=0.0001):
    self.map = map
    self.distance_penalty = distance_penalty
    self.angle_penalty = angle_penalty
    self.action_penalty = action_penalty

  def reward(self, s, a):
    d, delta_deg = self.map.distance_angle_deg(s[0], s[1], s[2])
    rda = self.reward_distance_angle(d, delta_deg)
    ra = self.reward_action(a)
    return rda + ra

  def reward_distance_angle(self, d, delta_deg):
    return -self.distance_penalty*d**2 - self.angle_penalty*delta_deg**2

  def reward_action(self, a_deg):
    return -self.action_penalty*a_deg**2

class LinearDistanceAngleReward:
  def __init__(self, map, distance_penalty=10.0, angle_penalty=0.05, action_penalty=0.1):
    self.map = map
    self.distance_penalty = distance_penalty
    self.angle_penalty = angle_penalty
    self.action_penalty = action_penalty

  def reward(self, s, a):
    d, delta_deg = self.map.distance_angle_deg(s[0], s[1], s[2])
    rda = self.reward_distance_angle(d, delta_deg)
    ra = self.reward_action(a)
    return rda + ra

  def reward_distance_angle(self, d, delta_deg):
    return -self.distance_penalty*abs(d) - self.angle_penalty*abs(delta_deg)

  def reward_action(self, a_deg):
    return -self.action_penalty*abs(a_deg)


class DrivingEnv:
  """State: [x, y, theta_deg]"""
  def __init__(self,
               map=make_oval(),
               car=DubinsCarModel(),
               dt=0.5,
               reward=LinearDistanceAngleReward(make_oval()).reward,
               init_state=np.array((1.0, 0.5, 0.0))):

    self.map = map
    self.car = car
    self.dt = dt
    self.reward = reward
    self.state = init_state

  def step(self, a_deg):
    r = self.reward(self.state, a_deg)
    state_rad = self.state*(1.0, 1.0, pi/180)
    state_rad = self.car.dynamics(state_rad, a_deg*pi/180, self.dt)
    self.state = state_rad*(1.0, 1.0, 180/pi)
    return self.state, r, False, None

  def reset(self):
    x, y = self.map.sample()
    self.state = np.array((x, y, 360*random()))
    return self.state
