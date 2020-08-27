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
	A drone (with yaw = 0) flies from the point above its takeoff point to the point above its landing spot
	The takeoff and landing operations are performed using airsim functions
	Data is logged once every self.save_log_steps timesteps

Source:
	This environment is as follows:

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

Reward:
	(i) At each step: inverse of 3D distance from the landing pad
	(ii) Inside Landing Pad: scale * inverse of height from the landing pad (scale = 2)
	(iii) Out of Bounds: -10
	(iv) Landed: 500/1000/10000
	(v) On landing, scale (iv) according to drone state (range: [0.5*landed, 1.5*landed])

Starting State:
	Starts at a random position (within a box defined in reset()), hovers, and takes off
"""

class AirSim(gym.Env):

	def __init__(self, name="Drone1", reward_landing=500, restrict=False):

		self.name = name
		self.restrict = restrict
		self.reward_landing = reward_landing

		# self.drone = ["Drone3", "Drone1", "Drone2"]
		# self.drone_offset_y = [-20, 0, 20]
		# self.drone_index = self.drone.index(self.name)
		# self.restrict = False
		# self.reward_landing = 500

		# connect to the AirSim simulator
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection() #confirms connection
		self.client.enableApiControl(True, vehicle_name=self.name) #Gives the control to the python script
		
		# logging data
		self.vehicle_has_collided = False
		self.timestamp = 0

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

		if self.restrict:
			# expert_traj_140_soft_sample_25_original - Preferably for human expert
			self.action_ub = np.array([0.25, 0.05, 0.675])
			self.action_lb = np.array([-0.25, -0.05, 0.30])
		else:
			# expert_traj_140_soft_sample_25_original - Preferably for RL expert
			self.action_ub = np.array([0.25, 0.05, 1])
			self.action_lb = np.array([-0.25, -0.05, 0])

		self.observation_ub = np.array([17, 2, 0.1, 3, 1, 4]) #self.observation_ub[0] = 11.5 for Bochan's data
		self.observation_lb = np.array([0, -2, -5, -1, -1, -4])

		# self.observation_ub = np.array([17, 2+self.drone_offset_y[self.drone_index], 0.1, 3, 1, 4]) #self.observation_ub[0] = 11.5 for Bochan's data
		# self.observation_lb = np.array([0, -2+self.drone_offset_y[self.drone_index], -5, -1, -1, -4])
		
		self.observation_space = spaces.Box(self.observation_lb, self.observation_ub, dtype=np.float32)
		self.action_space = spaces.Box(self.action_lb, self.action_ub, dtype=np.float32)

		# self.visual_cue_position = (13.5, 0, 0.3)
		self.visual_cue_height = 0.3
		self.target_size = 4 #1.75 for Bochan's data
		self.target_x, self.target_y, self.target_z = np.array([15, 0, -0.1]) #self.target_x = 10 for Bochan's data
		# self.target_x, self.target_y, self.target_z = np.array([15, 0+self.drone_offset_y[self.drone_index], -0.1]) #self.target_x = 10 for Bochan's data

	def reset(self, start):
		if not self.client.isApiControlEnabled():
			self.client.enableApiControl(True)
		# self.client.armDisarm(True)

		# ux = np.random.uniform(0, 4)
		# uy = np.random.uniform(-0.5, 0.5)
		# uz = np.random.uniform(-1.5, -2.5)
		# ux = 0
		# uy = 0
		# uz = -2

		import pdb; pdb.set_trace()

		ux = start[0].item()
		uy = start[1].item()
		uz = start[2].item()
		self.client.moveToPositionAsync(ux, uy, uz, 1).join()
		self.client.rotateToYawAsync(0, 1).join()

		self.total_steps = 0
		self.epi_t = 0
		self.episode += 1
		self.reward = 0
		self.count = 0

		# self.get_states()
		# return self.observations
		return np.asarray(start, dtype=np.float32)

	def normalize_obs(self):
		numerator_obs = np.subtract(self.states, self.observation_lb)
		denominator_obs = self.observation_ub - self.observation_lb
		states = numerator_obs / denominator_obs
		return states

	def get_states(self):
		lv = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.linear_velocity
		self.vx = lv.x_val
		self.vy = lv.y_val
		self.vz = lv.z_val
		pos = self.client.getMultirotorState(vehicle_name=self.name).kinematics_estimated.position
		self.x = pos.x_val
		self.y = pos.y_val
		self.z = pos.z_val

		self.vehicle_has_collided = self.client.simGetCollisionInfo(vehicle_name=self.name).has_collided
		self.timestamp = self.client.getMultirotorState(vehicle_name=self.name).timestamp
		self.states = (self.x, self.y, self.z, self.vx, self.vy, self.vz) #Observations

		# self.x, self.y, self.z = next[0].item(), next[1].item(), next[2].item()
		# self.vx, self.vy, self.vz = next[3].item(), next[4].item(), next[5].item()
		# self.states = (self.x, self.y, self.z, self.vx, self.vy, self.vz)

		self.observations = np.asarray(self.states, dtype=np.float32)

		# print('states', self.states)
		# print('observations', self.observations)

	def reward_function(self, current):
		# self.get_states()
		self.x, self.y, self.z = current[0].item(), current[1].item(), current[2].item()

		# if self.t < 5:
			# import pdb; pdb.set_trace()

		count = 0

		dist_to_pad_2d = np.sqrt( (self.x - self.target_x)**2 + (self.y - self.target_y)**2 )
		dist_to_pad_3d = np.sqrt( (self.x - self.target_x)**2 + (self.y - self.target_y)**2 + (self.z - self.target_z)**2 )

		# out_of_bounds = ( ( (self.observation_lb[0] <= self.states[0] <= self.observation_ub[0]) and (self.observation_lb[1] <= self.states[1] <= self.observation_ub[1]) and (self.observation_lb[2] <= self.states[2] <= self.observation_ub[2]) ) == False)
		out_of_bounds = ( ( (self.observation_lb[0] <= self.x <= self.observation_ub[0]) and (self.observation_lb[1] <= self.y <= self.observation_ub[1]) and (self.observation_lb[2] <= self.z <= self.observation_ub[2]) ) == False)

		# if self.t == 150:
		# 	import pdb; pdb.set_trace()

		# Reward structure
		if ( (dist_to_pad_2d < self.target_size/2) and (abs(self.z) <= abs(self.target_z)) ): #Inside landing pad and landed

			# if self.restrict:
			# 	# print('Successfully landed...', 'reward: {}'.format(self.reward_landing))
			# 	return self.reward_landing, 1
			# else:
			# 	if (-0.02 <= self.pitch <= 0.02):
			# 		count+=0.25
			# 	else:
			# 		count-=0.25
			# 	if (0 <= self.throttle <= 0.6):
			# 		count+=0.75
			# 	else:
			# 		count-=0.75
			# 	if (0 <= self.vx <= 1.5):
			# 		count+=2
			# 	else:
			# 		count-=2
			# 	if (0 <= self.vz <= 2.5):
			# 		count+=1
			# 	else:
			# 		count-=1
			# 	# print('Successfully landed...', 'reward: {}'.format((8+count)*0.125*self.reward_landing))
			# 	return (8+count)*0.125*self.reward_landing, 1

			# print('Successfully landed...', 'reward: {}'.format(self.reward_landing))
			return self.reward_landing, 1

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
		self.client.armDisarm(False, vehicle_name=self.name)
		self.client.enableApiControl(False, vehicle_name=self.name)
		# self.log_file.close()

	def step(self, action, current, next):
		self.t += 1
		self.epi_t += 1
		self.total_steps += 1

		self.get_states() #Only for My-AirSim-env (feed current state to get reward)
		# import pdb; pdb.set_trace()

		reward, done = self.reward_function(current)
		self.reward = reward

		print('\nTimestep ', self.t)
		print('current state: ', current[:3])
		print('current observation: ', self.observations[:3])
		print('actions: ', action)
		# print('reward: ', reward)
		# print('self.reward: ', self.reward)

		# self.client.moveByAngleThrottleAsync(float(self.pitch), float(self.roll), float(self.throttle), float(0), self.time_step, vehicle_name='').join()
		self.client.moveToPositionAsync(next[0].item(), next[1].item(), next[2].item(), 1).join()
		self.get_states()

		# import pdb; pdb.set_trace()
		print('next state: ', next[:3])
		print('next observation: ', self.observations[:3])

		if done == 1:
			done = True
			self.client.simPause(True)
			time.sleep(2)
			self.client.simPause(False)
			# self.client.hoverAsync().join #For straight maneuver
		elif done == 2:
			done = True
			self.count=1
		else:
			done = False

		if self.epi_t == 150:
			done = True

		# return self.observations, self.reward, done, {}
		return next, self.reward, done, {}