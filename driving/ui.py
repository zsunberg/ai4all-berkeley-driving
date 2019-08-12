# everything that the students interact with should go here
# all angles should be in degrees
from env import *

def run(policy, env=DrivingEnv()):
  """Display a real-time animation of the car moving around."""
  pass

def sim(policy, env=DrivingEnv()):
  """Return a SimResult"""
  pass
  
def sim_many(policy, env=DrivingEnv(), episode_length=1000, n_episodes=1000):
  """Return a list of SimResults"""
  pass

# train
