# everything that the students interact with should go here
# all angles should be in degrees
from driving.env import *
from driving.visualization import *
import math

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import DQN

import sys
import matplotlib.pyplot as plt


def env_constructor():
    return DrivingEnv()


# StudentQModel: Where the students will interact with the OpenAI DQN
class StudentQModel():
	def __init__(self, env, exploring_rate=0.3, 
				actions=(-35.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 35.0)):

		def env_constructor():
			return env

		self.env = env
		self.model = DQN('MlpPolicy',  DummyVecEnv([lambda: env_constructor()]), verbose=1, exploration_fraction=exploring_rate)
		self.n =0 # Counts how many times training has been called
		self.actions = actions
		self.cur_acts = dict() # This is the dict of Action: Q-Value

	def train(self, n_steps=20000, breaks=10000, model_name="neural_net"):
		# Print/Save after every 10,000 Steps
		div = n_steps//breaks
		run_nums = [breaks] * div
		final_run = n_steps - div*breaks

		# To Avoid Scope issues in callback, making them self. variables
		self.n = 0
		self.breaks = breaks
		self.model_name = model_name

		# Run Training for the Students
		self.model.learn(total_timesteps=n_steps, callback=self.train_callback)
		
		print("Finished " + model_name + "_" + str(self.n))
		plot = plot_episode_training(self.env, self.basicPolicy) # To make sure that the plot prints
		plt.show()

	# Callback called during training
	def train_callback(self, _locals, _globals):
		self.n += 1
		if self.n % self.breaks == 0:

			print("Finished " + self.model_name + "_" + str(self.n))
			plot = plot_episode_training(self.env, self.basicPolicy) # To make sure that the plot prints
			plt.show()

			self.model.save(self.model_name + "_" + str(self.n))

	def q_value(self, x,y,theta, a):
		q_values = self.model.step_model.step([[x,y,theta],])[1][0]
		self.cur_acts = dict(zip(self.actions, q_values))

		return self.cur_acts[a]

	def load(self, filename):
		self.model = DQN.load(filename, self.env.modelEnv)

	def basicPolicy(self, x,y,theta):
	    best_a = self.model.predict([x,y,theta])[0]
	    return self.env.actions[best_a]

def view_reward(env):
    return view_sa_func(env.reward, env)

def view_q_value(model):
    return view_sa_func(model.q_value, model.env)
