import json
import os
import sys
import tqdm

# def remove_files(root, index, items = {"bev":".png", "meta":".json", "rgb":".png", "supervision":".npy", "surroundings":".npy"}):
def remove_files(root, index, items = {"meta":".json", "supervision":".npy", "surroundings":".npy"}):
	# items = {"measurements":".json", "rgb_front":".png"}/
	# items = {"measurements":".json",}
	items = {"supervision":".npy", "surroundings":".npy", "measurements":".json"}
	sub_folders = list(os.listdir(root))
	for sub_folder in sub_folders:
		if len(list(os.listdir(os.path.join(root, sub_folder)))) == 0:
			break
		items[sub_folder] = "." + list(os.listdir(os.path.join(root, sub_folder)))[0].split(".")[-1]
	for k, v in items.items():
		data_folder = os.path.join(root, k)
		total_len = len(os.listdir(data_folder))
		for i in range(index, total_len):
			file_name = str(i).zfill(4) + v
			file_name = os.path.join(data_folder, file_name)
			os.remove(file_name)


if __name__ == '__main__':
	routes_type = ["short"]
	# towns = ["town05", "town06","town07", "town10"]
	# towns = ["town01_addition", "town02_val"]
	towns = ["town01_val", "town02_val", "town03"]

	# result_path = ""
	result_path = "/home/wyz/TCP"
	result_pattern = "data_collect_{}_results.json" # town
	result_pattern = "results_aggressive_data_collect_{}_results.json" # town
	# result_pattern = "{}_results_02.json" # town

	# data_path = ""
	data_path = "/home/wyz/TCP/data"

	# data_pattern = "{}_{}" # town, type
	# data_pattern = "data_collect_{}_results" # town, type
	data_pattern = "{}" # town, type
	# data_pattern = "{}_addition" # town, type

	for type in routes_type:
		print(type)
		for town in tqdm.tqdm(towns):
			# result_file = result_pattern.format(town, type)
			result_file = result_pattern.format(town)
			# data_folder = data_pattern.format(town, type)
			data_folder = data_pattern.format(town)
			data_folder = os.path.join(data_path, data_folder)

			# print("data_folder:", data_folder)
			sub_folders = os.listdir(data_folder)
			sub_folders = sorted(list(sub_folders))

			# read the record of each route
			with open(os.path.join(result_path, result_file), 'r') as f:
				records = json.load(f)
			records = records["_checkpoint"]["records"]
			# print("records:", records)

			for index, record in enumerate(records):
				route_data_folder = os.path.join(data_folder, sub_folders[index])
				total_length = len(os.listdir(os.path.join(route_data_folder, "measurements")))

				if record["scores"]["score_composed"] >= 100:
					continue
				# timeout or blocked, remove the last ones where the vehicle stops
				if len(record["infractions"]["route_timeout"]) > 0 or \
					len(record["infractions"]["vehicle_blocked"]) > 0:
						stop_index = 0
						for i in range(total_length-1, 0, -1):
							with open(os.path.join(route_data_folder, "measurements", str(i).zfill(4)) + ".json", 'r') as mf:
								speed = json.load(mf)["speed"]
								if speed > 0.1:
									stop_index = i
									break
						stop_index = min(total_length, stop_index + 20)
						remove_files(route_data_folder, stop_index)
				# collision or red-light
				elif len(record["infractions"]["red_light"]) > 0 or \
					len(record["infractions"]["collisions_pedestrian"]) > 0 or \
					len(record["infractions"]["collisions_vehicle"]) > 0 or \
					len(record["infractions"]["collisions_layout"]) > 0:
					stop_index = max(0, total_length-10)
					remove_files(route_data_folder, stop_index)
			
				


