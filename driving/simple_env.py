import gym
from gym.spaces import Box, Discrete
from math import sin
from matplotlib import pyplot as plt
from driving.car import DubinsCarModel
from driving.env import *
import random

class SimpleDrivingEnv(gym.Env):
  """State: [d, theta_deg]"""
  def __init__(self,
               actions=(-35.0, -10.0, 0.0, 10.0, 35.0),
               dt=0.5,
               speed = DubinsCarModel().speed,
               reward_model = LinearDistanceAngleReward(make_oval())):

    self.actions = actions
    self.dt = dt
    self.speed = speed
    self.reward_model = reward_model
    self.t = 0

  def step(self, action_choice):
    a_deg = self.actions[action_choice]
    return self.step_a_deg(a_deg)

  def step_a_deg(self, a_deg):
    d = self.state[0]
    theta_deg = self.state[1]
    r = self.reward_model.reward_distance_angle(d, theta_deg)
    r += self.reward_model.reward_action(a_deg)
    dp = d + sin(pi/180*theta_deg)*self.speed*self.dt
    theta_deg_p = theta_deg + a_deg*self.dt
    self.state = np.array((dp, theta_deg_p))
    if abs(dp) >= 10.0 or abs(theta_deg_p) >= 90.0:
        r -= 10.0
        done = True
    else:
        done = False
    self.t += 1
    if self.t > 100:
        done = True
    return self.state, r, done, dict()

  @property
  def observation_space(self):
    return Box(low=-90.0, high=90.0, shape=(2,))

  @property
  def action_space(self):
    return Discrete(5)

  def reset(self):
    d = random.random() - 0.5
    theta = (random.random() - 0.5)*15.0
    self.state = np.array((d, theta))
    self.t = 0
    return self.state

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

def plot_sim(env, policy, n_steps=100):
    history = sim(env, policy, n_steps)
    xs = range(len(history))
    ys = [step[0][0] for step in history]
    reward = sum([step[2] for step in history])
    print(f'reward: {reward}')
    plt.plot(xs, ys)
