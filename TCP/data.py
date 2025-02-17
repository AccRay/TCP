import os
from PIL import Image
import numpy as np
import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T


class CARLA_Data(Dataset):

	def __init__(self, root, data_folders, img_aug = False):
		"""
			root: root_dir_all
			data_folders: 
			img_aug:
		"""
		self.root = root
		self.img_aug = img_aug
		self._batch_read_number = 0

		self.front_img = []
		self.x = []
		self.y = []
		self.command = []
		self.target_command = []
		self.target_gps = []
		self.theta = []
		self.speed = []


		self.value = []
		self.feature = []
		self.action = []
		self.action_mu = []
		self.action_sigma = []

		self.future_x = []
		self.future_y = []
		self.future_theta = []

		self.future_feature = []
		self.future_action = []
		self.future_action_mu = []
		self.future_action_sigma = []
		self.future_only_ap_brake = []

		self.x_command = []
		self.y_command = []
		self.command = []
		self.only_ap_brake = []

		self.is_junction         = []
		self.traffic_light_state = []
		self.vehicles            = []
		self.walkers             = []
		self.stops               = []
		self.maximum_speed       = []
		self.stop_sign           = []
		self.yield_sign          = []
		self.wp = []

 
		for sub_root in data_folders:
			# need to check packed_data.npy
			data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()

			self.x_command += data['x_target']
			self.y_command += data['y_target']
			self.command += data['target_command']

			# self.front_img += data['front_img']
			self.x += data['input_x']
			self.y += data['input_y']
			self.theta += data['input_theta']
			self.speed += data['speed']

			self.future_x += data['future_x']
			self.future_y += data['future_y']
			self.future_theta += data['future_theta']

			self.future_feature += data['future_feature']
			self.future_action += data['future_action']
			self.future_action_mu += data['future_action_mu']
			self.future_action_sigma += data['future_action_sigma']
			self.future_only_ap_brake += data['future_only_ap_brake']

			self.value += data['value']
			self.feature += data['feature']
			self.action += data['action']
			self.action_mu += data['action_mu']
			self.action_sigma += data['action_sigma']
			self.only_ap_brake += data['only_ap_brake']
				
			self.is_junction         += data['input_is_junction']
			self.traffic_light_state += data['input_traffic_light_state']
			self.vehicles            += data['input_vehicles']
			self.walkers             += data['input_walkers']
			self.stops               += data['input_stops']
			self.maximum_speed       += data['input_maximum_speed']
			self.stop_sign           += data['input_stop_sign']
			self.yield_sign          += data['input_yield_sign']
			self.wp += data['wp']
      


		# It allows you to create a sequence of image transformations that can be applied
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

	def __len__(self):
		"""Returns the length of the dataset. """
		# return len(self.front_img)
		return len(self.is_junction)

	def __getitem__(self, index):
		"""Returns the item at index idx. """
		data = dict()

		# fix for theta=nan in some measurements
		if np.isnan(self.theta[index][0]):
			self.theta[index][0] = 0.

		ego_x = self.x[index][0]
		ego_y = self.y[index][0]
		ego_theta = self.theta[index][0]

		waypoints = []
		for i in range(4):
			R = np.array([      
			[np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
			[np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
			])
			local_command_point = np.array([self.future_y[index][i]-ego_y, self.future_x[index][i]-ego_x] )
			local_command_point = R.T.dot(local_command_point)
			waypoints.append(local_command_point)

		data['is_junction']         = torch.tensor(self.is_junction[index], dtype=torch.float32)
		# data['traffic_light_state'] = self.traffic_light_state[index]
		data['vehicles']	        = torch.tensor(flatten([item for sublist in self.vehicles[index] for item in sublist]), dtype=torch.float32)
		data['walkers']	            = torch.tensor(flatten([item for sublist in self.walkers[index] for item in sublist]), dtype=torch.float32)
		data['stops']               = torch.tensor(flatten(self.stops[index]), dtype=torch.float32)
		data['maximum_speed']       = torch.tensor([int(item) for item in self.maximum_speed[index]], dtype=torch.float32)
		data['stop_sign']           = torch.tensor(self.stop_sign[index], dtype=torch.float32)
		data['yield_sign']          = torch.tensor(self.yield_sign[index], dtype=torch.float32)
		wp = torch.squeeze(torch.tensor(self.wp[index], dtype=torch.float32), dim=0)
		data['wp'] = wp[:, 0:2].reshape(-1)

		data['waypoints'] = np.array(waypoints)

		data['action'] = self.action[index]
		data['action_mu'] = self.action_mu[index]
		data['action_sigma'] = self.action_sigma[index]


		future_only_ap_brake = self.future_only_ap_brake[index]
		future_action_mu = self.future_action_mu[index]
		future_action_sigma = self.future_action_sigma[index]

		# use the average value of roach braking action when the brake is only performed by the rule-based detector
		for i in range(len(future_only_ap_brake)):
			if future_only_ap_brake[i]:
				future_action_mu[i][0] = 0.8
				future_action_sigma[i][0] = 5.5
		data['future_action_mu'] = future_action_mu
		data['future_action_sigma'] = future_action_sigma
		data['future_feature'] = self.future_feature[index]

		only_ap_brake = self.only_ap_brake[index]
		if only_ap_brake:
			data['action_mu'][0] = 0.8
			data['action_sigma'][0] = 5.5

		R = np.array([
			[np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
			[np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
			])
		local_command_point = np.array([-1*(self.x_command[index]-ego_x), self.y_command[index]-ego_y] )
		local_command_point = R.T.dot(local_command_point)
		data['target_point'] = local_command_point[:2]


		local_command_point_aim = np.array([(self.y_command[index]-ego_y), self.x_command[index]-ego_x] )
		local_command_point_aim = R.T.dot(local_command_point_aim)
		data['target_point_aim'] = local_command_point_aim[:2]

		data['target_point'] = local_command_point_aim[:2]

		data['speed'] = self.speed[index]
		data['feature'] = self.feature[index]
		data['value'] = self.value[index]
		command = self.command[index]
		traffic_light_state = self.traffic_light_state[index]

		if traffic_light_state not in [0, 1, 2, 3]:
			traffic_light_state = 0
		assert  traffic_light_state in [0, 1, 2, 3]
		traffic_light_state_one_hot = [0] * 4
		traffic_light_state_one_hot[traffic_light_state] = 1
		data['traffic_light_state'] = torch.tensor(traffic_light_state_one_hot, dtype=torch.float32)

		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		data['target_command'] = torch.tensor(cmd_one_hot)		

		self._batch_read_number += 1
		return data

def flatten(input_list):
	result = []
	for item in input_list:
		if isinstance(item, list):
			result.extend(flatten(item))
		else:
			result.append(item)
	return result
