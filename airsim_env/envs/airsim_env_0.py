"""AirSim-v0 env for my Master's Thesis project on Autonomous UAV landing

- Author: Prabhasa Kalkur
- Contact: prabhasa@tamu.edu
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from datetime import datetime
import time, math
import os
import configparser
import sys
import numpy as np
import pandas as pd
import signal
from contextlib import contextmanager
import imutils
import airsim
import pdb

DEBUG_STATEMENTS = False

"""
Description:
	An OpenAI Gym env for autonomous UAV landing (binary built on Microsoft AirSim 2.0)
	A drone (with yaw = 0) flies from the point above its takeoff point to the point above its landing spot
	The takeoff and landing operations are performed using airsim functions
	Data is logged once every self.save_log_steps timesteps

Source:
	This environment is as follows:

Note: Bochan's binary has landing pad at (10, 0, -0.1) and PK's binary has landing pad at (15, 0, -0.1)

Observation (Bochan/PK):
	Type: Box(6)
	Num Observation             Min        Max
	0   Position, x            -1/0        11.5/17
	1   Position, y            -2          2
	2   Position, z            -4          0.1 (negative upwards)
	3   velocity, x            -1          4/3
	4   velocity, y            -1          1
	5   velocity, z            -2/-4       3/4
	
Actions (Bochan/PK):
	Type: Box(3)
	Num action                  Min         Max
	0   Pitch                  -0.3/-0.25   0.2 /0.25                   
	1   Roll                   -0.1/-0.05   0.1/0.05
	2   Throttle               0.55/0.3     0.8/0.675

Reward (simple):
	(i) At each step: inverse of 3D distance from the landing pad
	(ii) Inside Landing Pad: scale * inverse of height from the landing pad (scale = 2)
	(iii) Out of Bounds: -10
	(iv) Landed: 500
	(v) Reward (complex): On landing, scale (iv) according to drone state (slower speed => higher reward)

Starting State:
	Starts at a random position (within a box defined in reset()), hovers, and takes off
"""

class AirSim(gym.Env):

	def __init__(self, rew_complexity='simple', rew_land=1000, restrict=False):

		# NOTE (VERY IMPORTANT): These values are for the RL-generated expert data titled 'AirSim-v0_1.npz'
		# These can also be passed as env_kwargs()
		self.restrict = False 
		self.reward_landing = 1000 
		self.reward_complexity = 'simple'

		# NOTE: if rew_complexity == 'simple' (expert: 'AirSim-v0_1.npz') and 'complex' (expert: 'AirSim-v0_2.npz')
		if rew_complexity == 'simple':
			self.exp_id = 1
		else:
			self.exp_id = 2

		# connect to the AirSim simulator
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection() #confirms connection
		self.client.enableApiControl(True) #Gives the control to the python script

		# logging data
		self.vehicle_has_collided = False
		self.timestamp = 0

 		# Please change this as per your requirements
		self.name_folder = 'C:/IL_UAV/Files/logs'
		os.system('mkdir ' + self.name_folder)

		self.log_file = os.path.join(self.name_folder, 'log_airsim_v{}_{}.csv'.format(self.exp_id, datetime.now().strftime('%m_%d_%Y_%H_%M_%S')))

		self.episode = -1
		self.t = 0 #System clock
		self.epi_t = 0 #Episode clock
		self.count = 0 #Failure cases
		self.steps_beyond_done = None
		self.observations = None
		self.states = None
		self.reward = None

		self.time_step = 0.04 #1/sample_freq of expert
		self.save_log_steps = 1000
		self.reward_scaling = 2

		self._reset_log()
		self._save_log()

		if self.restrict:
			# Preferably for human expert
			self.action_ub = np.array([0.25, 0.05, 0.675])
			self.action_lb = np.array([-0.25, -0.05, 0.30])
		else:
			# Preferably for RL expert
			self.action_ub = np.array([0.25, 0.05, 1])
			self.action_lb = np.array([-0.25, -0.05, 0])

		self.observation_ub = np.array([17, 2, 0.1, 3, 1, 4]) #self.observation_ub[0] = 11.5 for Bochan's binary
		self.observation_lb = np.array([0, -2, -5, -1, -1, -4])
		
		self.observation_space = spaces.Box(self.observation_lb, self.observation_ub, dtype=np.float32)
		self.action_space = spaces.Box(self.action_lb, self.action_ub, dtype=np.float32)

		# Visual_cue_position = (18.5, 0, 0.3)
		self.visual_cue_height = 0.3
		self.target_size = 4 #1.75 for Bochan's binary
		self.target_x, self.target_y, self.target_z = np.array([15, 0, -0.1]) #self.target_x = 10 for Bochan's binary

	def reset(self):
		if not self.client.isApiControlEnabled():
			self.client.enableApiControl(True)
		# self.client.armDisarm(True)

		ux = np.random.uniform(0, 4)
		uy = np.random.uniform(-0.5, 0.5)
		uz = np.random.uniform(-2.5, -1.5)
		self.client.moveToPositionAsync(ux, uy, uz, 1).join()
		self.client.rotateToYawAsync(yaw=0).join()

		self.total_steps = 0
		self.epi_t = 0
		self.episode += 1
		self.reward = 0
		self.count = 0

		self.get_states()
		return self.observations

	def normalize_obs(self):
		numerator_obs = np.subtract(self.states, self.observation_lb)
		denominator_obs = self.observation_ub - self.observation_lb
		states = numerator_obs / denominator_obs
		return states

	def get_states(self):
		lv = self.client.getMultirotorState().kinematics_estimated.linear_velocity
		self.vx = lv.x_val
		self.vy = lv.y_val
		self.vz = lv.z_val
		pos = self.client.getMultirotorState().kinematics_estimated.position
		self.x = pos.x_val
		self.y = pos.y_val
		self.z = pos.z_val

		self.vehicle_has_collided = self.client.simGetCollisionInfo().has_collided
		self.timestamp = self.client.getMultirotorState().timestamp
		self.states = (self.x, self.y, self.z, self.vx, self.vy, self.vz) #Observations

		self.observations = np.asarray(self.states, dtype=np.float32)

	def reward_function(self):
		self.get_states()
		count = 0

		dist_to_pad_2d = np.sqrt( (self.x - self.target_x)**2 + (self.y - self.target_y)**2 )
		dist_to_pad_3d = np.sqrt( (self.x - self.target_x)**2 + (self.y - self.target_y)**2 + (self.z - self.target_z)**2 )

		out_of_bounds = ( ( (self.observation_lb[0] <= self.states[0] <= self.observation_ub[0]) and (self.observation_lb[1] <= self.states[1] <= self.observation_ub[1]) and (self.observation_lb[2] <= self.states[2] <= self.observation_ub[2]) ) == False)

		# Reward structure
		if ( (dist_to_pad_2d < self.target_size/2) and (abs(self.z) <= abs(self.target_z) or (self.vehicle_has_collided)) ): #Inside landing pad and landed
			
			if self.reward_complexity == 'simple':
				# print('Successfully landed...', 'reward: {}'.format(self.reward_landing))
				return self.reward_landing, 1
			else:
				if (-0.02 <= self.pitch <= 0.02):
					count+=0.25
				else:
					count-=0.25
				if (0 <= self.throttle <= 0.6):
					count+=0.75
				else:
					count-=0.75
				if (0 <= self.vx <= 1.5):
					count+=2
				else:
					count-=2
				if (0 <= self.vz <= 2.5):
					count+=1
				else:
					count-=1
				# print('Successfully landed...', 'reward: {}'.format((8+count)*0.125*self.reward_landing))
				return (8+count)*0.125*self.reward_landing, 1

		elif ( (dist_to_pad_2d < self.target_size/2) and (abs(self.z) > abs(self.target_z)) ):
			# print('Inside landing pad...')
			return abs(self.reward_scaling/self.z), 0
		else:                    
			if (out_of_bounds):
				# print('Out of bounds. Resetting...', 'reward: ', -10)
				return -10, 2
			elif ( (self.vehicle_has_collided) or (abs(self.z) < abs(self.visual_cue_height)) ):
				# print('Below visual cue. Resetting...', 'reward: ', -10)
				return -10, 2
			else:
				return  abs( (0.5*self.reward_scaling) / (dist_to_pad_3d) ), 0

	def render(self, mode='human'):
		pass

	def close(self):
		self.client.reset()
		self.client.armDisarm(False)
		self.client.enableApiControl(False)
		# self.log_file.close()

	def debug(self, *args, **kwargs):
		force = kwargs['force'] if 'force' in kwargs else False
		if DEBUG_STATEMENTS or force:
			print('DEBUG: {:s}'.format(*args))

	def _reset_log(self):
		self.column_names = ['id', 'episode', 'pitch', 'roll',
							 'throttle', 'reward', 'done', 'timestamp',
							 'failure_status', 'vehicle_pos_x', 'vehicle_pos_y',
							 'vehicle_pos_z', 'vehicle_vel_x', 'vehicle_vel_y',
							 'vehicle_vel_z']
		self.dataframe = {}
		for c in self.column_names:
			self.dataframe[c] = []

	def _save_log(self):
		if(self.t < 1):
			header = True
			mode = 'w'
		else:
			header = False
			mode = 'a'
		dataframe = pd.DataFrame(self.dataframe)
		dataframe.to_csv(self.log_file, header=header, mode=mode, index=False, line_terminator='\n')

	def _write_log(self, reward, done):
		self.dataframe[self.column_names[0]].append(self.t)
		self.dataframe[self.column_names[1]].append(self.episode)
		self.dataframe[self.column_names[2]].append(self.pitch) 
		self.dataframe[self.column_names[3]].append(self.roll)
		self.dataframe[self.column_names[4]].append(self.throttle)
		self.dataframe[self.column_names[5]].append(reward)
		self.dataframe[self.column_names[6]].append(done)
		self.dataframe[self.column_names[7]].append(self.timestamp)
		# self.dataframe[self.column_names[8]].append(1 if self.vehicle_has_collided else 0)
		self.dataframe[self.column_names[8]].append(self.count)
		self.dataframe[self.column_names[9]].append(self.x)
		self.dataframe[self.column_names[10]].append(self.y)
		self.dataframe[self.column_names[11]].append(self.z)
		self.dataframe[self.column_names[12]].append(self.vx)
		self.dataframe[self.column_names[13]].append(self.vy)
		self.dataframe[self.column_names[14]].append(self.vz)

	def step(self, action):
		self.t += 1
		self.epi_t += 1
		self.total_steps += 1

		action_org = np.clip(action, self.action_lb, self.action_ub)
		
		self.pitch = action_org[0]
		self.roll = action_org[1]
		self.throttle = action_org[2]

		reward, done = self.reward_function()
		self.reward = np.float64(reward)

		self.client.moveByAngleThrottleAsync(float(self.pitch), float(self.roll), float(self.throttle), float(0), duration=self.time_step).join() #default vehicle_name=''
		
		self.get_states()

		if done == 1:
			done = True
			self.client.simPause(True)
			time.sleep(2)
			self.client.simPause(False)
		elif done == 2:
			done = True
			self.count=1
		else:
			done = False

		self._write_log(reward, done)
		if(self.t % self.save_log_steps == 0):
			self._save_log()
			self._reset_log()

		if self.epi_t == 150:# Depends on expert data avg length
			done = True

		return self.observations, self.reward, done, {}