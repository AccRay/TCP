from distutils.log import error
import os
import json
from typing import DefaultDict
import numpy as np
import tqdm

from multiprocessing import Pool


INPUT_FRAMES = 1
FUTURE_FRAMES = 4

MAX_ACTORS = 3

def process_actor_info(actor_list, direction):
	"""
	Delete farthest vehicles with a list quantity exceeding 3
	Fill in 0 if the list quantity is less than 3
	remove keys from dic
	"""
	training_list = []
	if len(actor_list) > 3:
		sorted_dicts = sorted(actor_list, key=lambda x: x["distance_to_ego"])
		actor_list = sorted_dicts[:3]
	if len(actor_list) > 0:
		for actor in actor_list:
			distance_to_ego  = actor['distance_to_ego']
			direction_to_ego = actor['direction_to_ego']
			speed            = actor['speed']
			forward_vector   = actor['forward_vector']
			training_list.append([distance_to_ego, direction_to_ego, speed, forward_vector, direction])
	training_list = training_list + [[0, 0, 0, 0, 0]] * (3 - len(training_list))
	return training_list

def gen_single_route(route_folder):

	length = len(os.listdir(os.path.join(route_folder, 'measurements')))
	# print(os.listdir(os.path.join(route_folder, 'measurements'))) # json01-json17
	# exit()
	if length < INPUT_FRAMES + FUTURE_FRAMES:
		return

	seq_future_x = []
	seq_future_y = []
	seq_future_theta = []
	seq_future_feature = []
	seq_future_action = []
	seq_future_action_mu = []
	seq_future_action_sigma = []
	seq_future_only_ap_brake = []


	seq_input_x = []
	seq_input_y = []
	seq_input_theta = []

	seq_input_is_junction         = []
	seq_input_traffic_light_state = []
	# seq_input_left_vehicles       = []
	# seq_input_mid_vehicles        = []
	# seq_input_right_vehicles      = []
	seq_input_vehicles            = []
	# seq_input_left_walkers        = []
	# seq_input_mid_walkers         = []
	# seq_input_right_walkers       = []
	seq_input_walkers             = []
	seq_input_stops               = []
	seq_input_maximum_speed       = []
	seq_input_stop_sign           = []
	seq_input_yield_sign          = []

	seq_input_wp = []
	full_seq_wp = []





	seq_front_img = []
	seq_feature = []
	seq_value = []
	seq_speed = []

	seq_action = []
	seq_action_mu = []
	seq_action_sigma = []

	seq_x_target = []
	seq_y_target = []
	seq_target_command = []

	seq_only_ap_brake = []

	full_seq_x = []
	full_seq_y = []
	full_seq_theta = []

	full_seq_feature = []
	full_seq_action = []
	full_seq_action_mu = []
	full_seq_action_sigma = []
	full_seq_only_ap_brake = []


	seq_is_junction         = []
	seq_traffic_light_state = []
	seq_left_vehicles       = []
	seq_mid_vehicles        = []
	seq_right_vehicles      = []
	seq_left_walkers        = []
	seq_mid_walkers         = []
	seq_right_walkers       = []
	seq_stops               = []
	seq_maximum_speed       = []
	seq_stop_sign           = []
	seq_yield_sign          = []

	full_seq_is_junction         = []
	full_seq_traffic_light_state = []
	# full_seq_left_vehicles       = []
	# full_seq_mid_vehicles        = []
	# full_seq_right_vehicles      = []
	full_seq_vehicles            = []
	# full_seq_left_walkers        = []
	# full_seq_mid_walkers         = []
	# full_seq_right_walkers       = []
	full_seq_walkers       = []
	full_seq_stops               = []
	full_seq_maximum_speed       = []
	full_seq_stop_sign           = []
	full_seq_yield_sign          = []

	for i in range(length):
		with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
			measurement = json.load(read_file)
			full_seq_x.append(measurement['y'])
			full_seq_y.append(measurement['x'])
			full_seq_theta.append(measurement['theta'])


		roach_supervision_data = np.load(os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		full_seq_feature.append(roach_supervision_data['features'])
		full_seq_action.append(roach_supervision_data['action'])
		full_seq_action_mu.append(roach_supervision_data['action_mu'])
		full_seq_action_sigma.append(roach_supervision_data['action_sigma'])
		full_seq_only_ap_brake.append(roach_supervision_data['only_ap_brake'])

		# with open(os.path.join(route_folder, "surroundings", f"{str(i).zfill(4)}.json"), "r", encoding='utf-8') as read_file:
		# 	surroundings = json.load(read_file)
		# surroundings = np.load(os.path.join(route_folder, "surroundings", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		# print(process_actor_info(surroundings['mid_vehicles'], 2))
		# surroundings = read_file.read()
		try:
			surroundings = np.load(os.path.join(route_folder, "surroundings", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		except EOFError:
			print(f"Error reading file: {str(i).zfill(4)}.npy. It might be empty or corrupted.")
    # Handle the error, maybe skip this file or use a default value for `surroundings`
		surroundings_left_vehicles  = process_actor_info(surroundings['left_vehicles'], 1)
		surroundings_mid_vehicles   = process_actor_info(surroundings['mid_vehicles'], 2)
		surroundings_right_vehicles = process_actor_info(surroundings['right_vehicles'], 3)
		surroundings_left_walkers   = process_actor_info(surroundings['left_walkers'], 1)
		surroundings_mid_walkers    = process_actor_info(surroundings['mid_walkers'], 2)
		surroundings_right_walkers  = process_actor_info(surroundings['right_walkers'], 3)

		full_seq_is_junction.append(surroundings['is_junction'])
		full_seq_traffic_light_state.append(surroundings['traffic_light_state'])
		# full_seq_left_vehicles.append(surroundings_left_vehicles)
		# full_seq_mid_vehicles.append(surroundings_mid_vehicles)
		# full_seq_right_vehicles.append(surroundings_right_vehicles)
		full_seq_vehicles.append(surroundings_left_vehicles + surroundings_mid_vehicles + surroundings_right_vehicles)
		# full_seq_left_walkers.append(surroundings_left_walkers)
		# full_seq_mid_walkers.append(surroundings_mid_walkers)
		# full_seq_right_walkers.append(surroundings_right_walkers)
		full_seq_walkers.append(surroundings_left_walkers + surroundings_mid_walkers + surroundings_right_walkers)
		full_seq_stops.append(surroundings['stops'])
		full_seq_maximum_speed.append(surroundings['MaximumSpeed'])
		full_seq_stop_sign.append(surroundings['StopSign'])
		full_seq_yield_sign.append(surroundings['YieldSign'])
		full_seq_wp.append(surroundings['wp'])
	# todo 
	# max vehicles(left 3, right 3, mid 3)
	# 
	# print(full_seq_is_junction)
	# print(full_seq_traffic_light_state)
	# print(full_seq_left_vehicles)
	# print(full_seq_vehicles)
	# print(full_seq_right_vehicles)
	# print(full_seq_left_walkers)
	# print(full_seq_mid_walkers)
	# print(full_seq_right_walkers)
	# print(full_seq_stops)
	# print(full_seq_maximum_speed)
	# print(full_seq_stop_sign)
	# print(full_seq_yield_sign)
	# exit()
	
	for i in range(INPUT_FRAMES-1, length-FUTURE_FRAMES):

		with open(os.path.join(route_folder, "measurements", f"{str(i).zfill(4)}.json"), "r") as read_file:
			measurement = json.load(read_file)
		# 0 - (n-5)
		seq_input_x.append(full_seq_x[i-(INPUT_FRAMES-1):i+1])
		seq_input_y.append(full_seq_y[i-(INPUT_FRAMES-1):i+1])
		seq_input_theta.append(full_seq_theta[i-(INPUT_FRAMES-1):i+1])

		# add surroundings (compare the input of the image)
		seq_input_is_junction.append(full_seq_is_junction[i-(INPUT_FRAMES-1):i+1])
		seq_input_traffic_light_state.append(full_seq_traffic_light_state[i-(INPUT_FRAMES-1):i+1])
		# seq_input_left_vehicles.append(full_seq_left_vehicles[i-(INPUT_FRAMES-1):i+1])
		# seq_input_mid_vehicles.append(full_seq_mid_vehicles[i-(INPUT_FRAMES-1):i+1])
		# seq_input_right_vehicles.append(full_seq_right_vehicles[i-(INPUT_FRAMES-1):i+1])
		seq_input_vehicles.append(full_seq_vehicles[i-(INPUT_FRAMES-1):i+1])

		# seq_input_left_walkers.append(full_seq_left_walkers[i-(INPUT_FRAMES-1):i+1])
		# seq_input_mid_walkers.append(full_seq_mid_walkers[i-(INPUT_FRAMES-1):i+1])
		# seq_input_right_walkers.append(full_seq_right_walkers[i-(INPUT_FRAMES-1):i+1])
		seq_input_walkers.append(full_seq_walkers[i-(INPUT_FRAMES-1):i+1])

		seq_input_stops.append(full_seq_stops[i-(INPUT_FRAMES-1):i+1])
		seq_input_maximum_speed.append(full_seq_maximum_speed[i-(INPUT_FRAMES-1):i+1])
		seq_input_stop_sign.append(full_seq_stop_sign[i-(INPUT_FRAMES-1):i+1])
		seq_input_yield_sign.append(full_seq_yield_sign[i-(INPUT_FRAMES-1):i+1])
		seq_input_wp.append(full_seq_wp[i-(INPUT_FRAMES-1):i+1])



		# i+1 - i+5
		seq_future_x.append(full_seq_x[i+1:i+FUTURE_FRAMES+1])
		seq_future_y.append(full_seq_y[i+1:i+FUTURE_FRAMES+1])
		seq_future_theta.append(full_seq_theta[i+1:i+FUTURE_FRAMES+1])

		seq_future_feature.append(full_seq_feature[i+1:i+FUTURE_FRAMES+1])
		seq_future_action.append(full_seq_action[i+1:i+FUTURE_FRAMES+1])
		seq_future_action_mu.append(full_seq_action_mu[i+1:i+FUTURE_FRAMES+1])
		seq_future_action_sigma.append(full_seq_action_sigma[i+1:i+FUTURE_FRAMES+1])
		seq_future_only_ap_brake.append(full_seq_only_ap_brake[i+1:i+FUTURE_FRAMES+1])

		roach_supervision_data = np.load(os.path.join(route_folder, "supervision", f"{str(i).zfill(4)}.npy"), allow_pickle=True).item()
		seq_feature.append(roach_supervision_data["features"])
		seq_value.append(roach_supervision_data["value"])
		
		# use to store img
		# front_img_list = [route_folder.replace(data_path,'')+"/rgb/"f"{str(i-_).zfill(4)}.png" for _ in range(INPUT_FRAMES-1, -1, -1)]
		# i = 3 , 0001.png,0002.png, 0003.png
		# seq_front_img.append(front_img_list)
		# we use surroundings instead of img

		seq_speed.append(measurement["speed"])

		seq_action.append(roach_supervision_data["action"])
		seq_action_mu.append(roach_supervision_data["action_mu"])
		seq_action_sigma.append(roach_supervision_data["action_sigma"])

		seq_x_target.append(measurement["y_target"])
		seq_y_target.append(measurement["x_target"])
		seq_target_command.append(measurement["target_command"])

		seq_only_ap_brake.append(roach_supervision_data["only_ap_brake"])

	return seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_feature, seq_value, seq_speed, seq_action, seq_action_mu, seq_action_sigma, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake, seq_input_is_junction, seq_input_traffic_light_state, seq_input_vehicles, seq_input_walkers, seq_input_stops, seq_input_maximum_speed, seq_input_stop_sign, seq_input_yield_sign, seq_input_wp



def gen_sub_folder(folder_path):
	route_list = [folder for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]
	route_list = sorted(route_list)

	total_future_x = []
	total_future_y = []
	total_future_theta = []

	total_future_feature = []
	total_future_action = []
	total_future_action_mu = []
	total_future_action_sigma = []
	total_future_only_ap_brake = []

	total_input_x = []
	total_input_y = []
	total_input_theta = []

	total_front_img = []
	total_feature = []
	total_value = []
	total_speed = []

	total_action = []
	total_action_mu = []
	total_action_sigma = []

	total_x_target = []
	total_y_target = []
	total_target_command = []

	total_only_ap_brake = []

	total_input_is_junction         = []
	total_input_traffic_light_state = []
	total_input_vehicles            = []
	total_input_walkers             = []
	total_input_stops               = []
	total_input_maximum_speed       = []
	total_input_stop_sign           = []
	total_input_yield_sign          = []
	total_input_wp = []



	for route in route_list:
		seq_data = gen_single_route(os.path.join(folder_path, route))
		if not seq_data:
			continue
		seq_future_x, seq_future_y, seq_future_theta, seq_future_feature, seq_future_action, seq_future_action_mu, seq_future_action_sigma, seq_future_only_ap_brake, seq_input_x, seq_input_y, seq_input_theta, seq_feature, seq_value, seq_speed, seq_action, seq_action_mu, seq_action_sigma, seq_x_target, seq_y_target, seq_target_command, seq_only_ap_brake, seq_input_is_junction, seq_input_traffic_light_state, seq_input_vehicles, seq_input_walkers, seq_input_stops, seq_input_maximum_speed, seq_input_stop_sign, seq_input_yield_sign, seq_input_wp = seq_data
		
		total_future_x.extend(seq_future_x)
		total_future_y.extend(seq_future_y)
		total_future_theta.extend(seq_future_theta)
		total_future_feature.extend(seq_future_feature)
		total_future_action.extend(seq_future_action)
		total_future_action_mu.extend(seq_future_action_mu)
		total_future_action_sigma.extend(seq_future_action_sigma)
		total_future_only_ap_brake.extend(seq_future_only_ap_brake)
		total_input_x.extend(seq_input_x)
		total_input_y.extend(seq_input_y)
		total_input_theta.extend(seq_input_theta)

		total_input_is_junction.extend(seq_input_is_junction)
		total_input_traffic_light_state.extend(seq_input_traffic_light_state)
		total_input_vehicles.extend(seq_input_vehicles)
		total_input_walkers.extend(seq_input_walkers)
		total_input_stops.extend(seq_input_stops)
		total_input_maximum_speed.extend(seq_input_maximum_speed)
		total_input_stop_sign.extend(seq_input_stop_sign)
		total_input_yield_sign.extend(seq_input_yield_sign)
		total_input_wp.extend(seq_input_wp)



		# total_front_img.extend(seq_front_img)
		total_feature.extend(seq_feature)
		total_value.extend(seq_value)
		total_speed.extend(seq_speed)
		total_action.extend(seq_action)
		total_action_mu.extend(seq_action_mu)
		total_action_sigma.extend(seq_action_sigma)
		total_x_target.extend(seq_x_target)
		total_y_target.extend(seq_y_target)
		total_target_command.extend(seq_target_command)
		total_only_ap_brake.extend(seq_only_ap_brake)

	data_dict = {}
	data_dict['future_x'] = total_future_x
	data_dict['future_y'] = total_future_y
	data_dict['future_theta'] = total_future_theta
	data_dict['future_feature'] = total_future_feature
	data_dict['future_action'] = total_future_action
	data_dict['future_action_mu'] = total_future_action_mu
	data_dict['future_action_sigma'] = total_future_action_sigma
	data_dict['future_only_ap_brake'] = total_future_only_ap_brake
	data_dict['input_x'] = total_input_x
	data_dict['input_y'] = total_input_y
	data_dict['input_theta'] = total_input_theta

	# data_dict['front_img'] = total_front_img
	data_dict['input_is_junction'] = total_input_is_junction
	data_dict['input_traffic_light_state'] = total_input_traffic_light_state
	data_dict['input_vehicles'] = total_input_vehicles
	data_dict['input_walkers'] = total_input_walkers
	data_dict['input_stops'] = total_input_stops
	data_dict['input_maximum_speed'] = total_input_maximum_speed
	data_dict['input_stop_sign'] = total_input_stop_sign
	data_dict['input_yield_sign'] = total_input_yield_sign
	data_dict['wp'] = total_input_wp


	data_dict['feature'] = total_feature
	data_dict['value'] = total_value
	data_dict['speed'] = total_speed
	data_dict['action'] = total_action
	data_dict['action_mu'] = total_action_mu
	data_dict['action_sigma'] = total_action_sigma
	data_dict['x_target'] = total_x_target
	data_dict['y_target'] = total_y_target
	data_dict['target_command'] = total_target_command
	data_dict['only_ap_brake'] = total_only_ap_brake

	file_path = os.path.join(folder_path, "packed_data")
	np.save(file_path, data_dict)
	return len(total_future_x)


if __name__ == '__main__':
	global data_path
	# data_path = "tcp_carla_data"
	data_path = "/home/wyz/TCP/data"
	# towns = ["town01","town01_val","town01_addition","town02","town02_val","town03","town03_val","town03_addition", "town04","town04_val", "town04_addition", "town05", "town05_val", "town05_addition" ,"town06","town06_val", "town06_addition","town07", "town07_val", "town10", "town10_addition","town10_val"]
	# towns = ["town01", "town01_addition"]
	towns = ["town04_val"]
	pattern = "{}" # town type
	import tqdm
	total = 0
	for town in tqdm.tqdm(towns):
		number = gen_sub_folder(os.path.join(data_path, pattern.format(town)))
		total += number

	print(total)

	