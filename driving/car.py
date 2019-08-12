import numpy as np
from math import pi, cos, sin

# all angles in this file are in RADIANS unless otherwise noted

class DubinsCarModel:
  """State: [x, y, theta_rad]"""

  def __init__(self, speed=0.3, max_turn_rate=90*pi/180):
    self.speed = speed
    self.max_turn_rate = max_turn_rate

  def dynamics(self, s, a, dt):
    """Calculate the new state given the previous state and action."""
    tr = np.clip(a, -self.max_turn_rate, self.max_turn_rate)
    x, y, theta = s
    thetap = theta + tr*dt
    while thetap > 2*pi:
        thetap -= 2*pi
    while thetap < 0.0:
        thetap += 2*pi
    eps = 1e-5
    if abs(tr) <= eps:
      xp = x + self.speed*cos(theta)*dt
      yp = y + self.speed*sin(theta)*dt
    else:
      xp = x + self.speed*(sin(thetap) - sin(theta))/tr
      yp = y - self.speed*(cos(thetap) - cos(theta))/tr
    return np.array((xp, yp, thetap))
