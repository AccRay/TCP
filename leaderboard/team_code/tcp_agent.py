import os
import json
import datetime
import pathlib
import time
import cv2
from collections import deque
import math
from collections import OrderedDict

import torch
import carla
import numpy as np
from torchvision import transforms as T
from leaderboard.autoagents import autonomous_agent

from TCP.model_transformerV3 import TCP
from TCP.config import GlobalConfig
from team_code.planner import RoutePlanner
from leaderboard.utils.route_manipulation import downsample_route

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from roach.obs_manager.birdview.tcp_noming import ObsManager
from roach.utils.traffic_light import TrafficLightHandler
from omegaconf import OmegaConf
from roach.criteria import run_stop_sign
from agents.navigation.local_planner import RoadOption

SAVE_PATH = os.environ.get('SAVE_PATH', None)

TRAFFIC_LIGHT_STATE = {
    'None': 0,
    'Red': 1,
    'Green': 2,
    'Yellow': 3,
}


def get_entry_point():
    return 'TCPAgent'


class TCPAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.velocity_sum = 0
        self.Jerk_sum = 0
        self.TTC = 100000
        self.imp = np.array([])
        self.last_acc = 0
        self.lane_diff_sum = 0
        self.wall_start = time.time()
        self.initialized = False

        self.config = GlobalConfig()
        self.net = TCP(self.config)

        # we need a ckpt model
        # print("************************************")
        # print(path_to_conf_file)
        # print("************************************")
        ckpt = torch.load(path_to_conf_file, map_location='cuda:0')
        # ckpt = torch.load(path_to_conf_file)

        ckpt = ckpt["state_dict"]
        new_state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value
        self.net.load_state_dict(new_state_dict, strict=False)
        self.net.cuda()
        self.net.eval()

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None
        self._im_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.last_steers = deque()

        cfg = OmegaConf.load('./roach/config/config_agent.yaml')
        cfg = OmegaConf.to_container(cfg)
        self.cfg = cfg

        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()

            (self.save_path / 'surroundings').mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 48.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

        self._ego_vehicle = CarlaDataProvider.get_ego()
        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()
        self._criteria_stop = run_stop_sign.RunStopSign(self._world)
        self.birdview_obs_manager = ObsManager(
            self.cfg['obs_configs']['birdview'], self._criteria_stop)
        self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)
        TrafficLightHandler.reset(self._world)

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': -1.5, 'y': 0.0, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 900, 'height': 256, 'fov': 100,
                'id': 'rgb'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'bev'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]

    def tick(self, input_data):
        self._truncate_global_route_till_local_target()
        # self._draw_waypoints(CarlaDataProvider.get_world(), self._global_route, vertical_shift=1.0, persistency=1)
        # print(len(self._global_route))
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        # birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)
        surroundings_info = self.birdview_obs_manager.get_surroundings()

        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            'rgb': rgb,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'bev': bev
        }

        pos = self._get_position(result)
        result['gps'] = pos

        next_wp, next_cmd = self._route_planner.run_step(pos)

        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)
        if (result['target_point'][0]) > 1 and abs(result['target_point'][1]) > 1:
            self._route_planner.max_distance = 16
            next_wp, next_cmd = self._route_planner.run_step(pos)
            result['next_command'] = next_cmd.value
            theta = compass + np.pi / 2
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
            local_command_point = R.T.dot(local_command_point)
            result['target_point'] = tuple(local_command_point)
            self._route_planner.max_distance = 48

        return result, surroundings_info

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        ev_location = self._ego_vehicle.get_location()
        wp_end = []
        if (len(self._global_route) >= 16):
            waypoints = self._global_route[0:16]

            for waypoint in waypoints:
                wp = waypoint[0].transform.location - ev_location
                wp_end.append([wp.x, wp.y, wp.z])
        else:
            waypoints = self._global_route[0:len(self._global_route)]
            for waypoint in waypoints:
                wp = waypoint[0].transform.location - ev_location
                wp_end.append([wp.x, wp.y, wp.z])
            while len(wp_end) < 16:
                wp_end.append([wp.x, wp.y, wp.z])

        wp_temp = torch.tensor(wp_end, dtype=torch.float32)
        waypoints_end = wp_temp[:, 0:2].reshape(-1).view(1, -1).to('cuda')

        tick_data, surroundings_info = self.tick(input_data)

        surroundings_left_vehicles = process_actor_info(surroundings_info['left_vehicles'], 1)
        surroundings_mid_vehicles = process_actor_info(surroundings_info['mid_vehicles'], 2)
        surroundings_right_vehicles = process_actor_info(surroundings_info['right_vehicles'], 3)
        surroundings_back_vehicles = process_actor_info(surroundings_info['back_vehicles'], 4)
        surroundings_left_walkers = process_actor_info(surroundings_info['left_walkers'], 1)
        surroundings_mid_walkers = process_actor_info(surroundings_info['mid_walkers'], 2)
        surroundings_right_walkers = process_actor_info(surroundings_info['right_walkers'], 3)
        ego_vehicle_waypoint = self._map.get_waypoint(self._ego_vehicle.get_location())
        vehicles_temp = surroundings_left_vehicles + surroundings_mid_vehicles + surroundings_right_vehicles
        walkers_temp = surroundings_left_walkers + surroundings_mid_walkers + surroundings_right_walkers
        is_junction_temp = 1 if ego_vehicle_waypoint.is_junction else 0
        traffic_light_state_temp, light_loc, light_id = TrafficLightHandler.get_light_state(self._ego_vehicle)

        traffic_light_state = TRAFFIC_LIGHT_STATE[str(traffic_light_state_temp)]
        is_junction = torch.tensor(is_junction_temp, dtype=torch.float32).view(1, 1).to('cuda')
        if traffic_light_state not in [0, 1, 2, 3]:
            traffic_light_state = 0
        assert traffic_light_state in [0, 1, 2, 3]
        traffic_light_state_one_hot = [0] * 4
        # if traffic_light_state == 1 or traffic_light_state == 3: # 测试
        # 	traffic_light_state = 2 # 测试
        traffic_light_state_one_hot[traffic_light_state] = 1
        traffic_light_state_end = torch.tensor(traffic_light_state_one_hot, dtype=torch.float32).view(1, -1).to('cuda')
        stops = torch.tensor(flatten(surroundings_info['stops']), dtype=torch.float32).view(1, -1).to('cuda')
        max_speed = torch.tensor(int(surroundings_info['MaximumSpeed']), dtype=torch.float32).view(1, 1).to('cuda')
        stop_sign = torch.tensor(surroundings_info['StopSign'], dtype=torch.float32).view(1, -1).to('cuda')
        yield_sign = torch.tensor(surroundings_info['YieldSign'], dtype=torch.float32).view(1, -1).to('cuda')
        vehicles = torch.tensor(flatten([item for sublist in vehicles_temp for item in sublist]),
                                dtype=torch.float32).view(1, -1).to('cuda')
        walkers = torch.tensor(flatten([item for sublist in walkers_temp for item in sublist]),
                               dtype=torch.float32).view(1, -1).to('cuda')

        if self.step < self.config.seq_len:
            # rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        # if waypoints_end[31] > 5 and
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        speed = torch.FloatTensor([float(tick_data['speed'])]).view(1, 1).to('cuda', dtype=torch.float32)
        speed = speed / 12
        # rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                     torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        state = torch.cat([speed, target_point, cmd_one_hot], 1)

        pred = self.net(is_junction, vehicles, walkers, stops, max_speed, stop_sign, yield_sign,
                        traffic_light_state_end, state, target_point, waypoints_end)

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'],
                                                                                  gt_velocity, target_point)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity,
                                                                                    target_point)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        self.pid_metadata = metadata_traj
        control = carla.VehicleControl()
        # print(self.status)

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            control.steer = np.clip(self.alpha * steer_ctrl + (1 - self.alpha) * steer_traj, -1, 1)
            control.throttle = np.clip(self.alpha * throttle_ctrl + (1 - self.alpha) * throttle_traj, 0, 0.75)
            control.brake = np.clip(self.alpha * brake_ctrl + (1 - self.alpha) * brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            control.steer = np.clip(self.alpha * steer_traj + (1 - self.alpha) * steer_ctrl, -1, 1)
            control.throttle = np.clip(self.alpha * throttle_traj + (1 - self.alpha) * throttle_ctrl, 0, 0.75)
            control.brake = np.clip(self.alpha * brake_traj + (1 - self.alpha) * brake_ctrl, 0, 1)

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if control.brake > 0.5:
            control.throttle = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(control.steer)))
        # chech whether ego is turning
        # num of steers larger than 0.1
        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1

        else:
            self.status = 0

        self.pid_metadata['status'] = self.status

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)

        # print(control.steer, control.throttle, control.brake, traffic_light_state_temp)
        #####################################evlautate#################################################################
        front_vehicle_info = surroundings_mid_vehicles[0]
        # TTC
        if front_vehicle_info[3] < 10 and front_vehicle_info[4] != 0:
            speed_diff = CarlaDataProvider.get_velocity(self._ego_vehicle) - front_vehicle_info[2]
            if speed_diff > 0:
                TTC = front_vehicle_info[0] / speed_diff
                self.TTC = min(self.TTC, TTC)
        # print(TTC)
        # velocity
        self.velocity_sum += CarlaDataProvider.get_velocity(self._ego_vehicle)
        # print(self.velocity_sum / self.step)
        # Jerk
        acc = self._ego_vehicle.get_acceleration()
        curr_acc = math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)
        self.Jerk_sum += abs(curr_acc - self.last_acc) / 0.05
        self.last_acc = curr_acc
        # print(self.Jerk_sum / self.step)
        # Impact on rear vehicle
        # print(surroundings_back_vehicles)
        if control.brake > 0:
            back_vehicle_info = surroundings_back_vehicles[0]
            if back_vehicle_info[3] < 10 and back_vehicle_info[4] != 0:
                self.imp = np.append(self.imp, back_vehicle_info[2])
            # if self.imp.shape[0] != 0:
            # 	print(self.imp.mean())
        # Deviation
        lane_center = self._map.get_waypoint(self._ego_vehicle.get_location())
        lane_diff = lane_center.transform.location.distance(self._ego_vehicle.get_location())
        # print(lane_diff)
        self.lane_diff_sum += lane_diff
        # print(self.lane_diff_sum / self.step)

        ###############################################################################################################
        front_walkers_info = surroundings_mid_walkers[0]
        right_walkers_info = surroundings_right_walkers[0]
        left_walkers_info = surroundings_left_walkers[0]
        if traffic_light_state == 2 or len(self._global_route) < 16:
            if front_vehicle_info[0] > 10 or front_vehicle_info[0] == 0:
                control.throttle = 0.75
                control.brake = 0
        elif traffic_light_state == 1 or traffic_light_state == 3:
            control.throttle = 0
            control.brake = 0.75
        if control.throttle < control.brake:
            if (traffic_light_state == 0 and front_vehicle_info[0] == 0 and front_walkers_info[0] == 0
                    and right_walkers_info[0] == 0 and left_walkers_info[0] == 0 and surroundings_left_vehicles[0][0] == 0
                    and surroundings_left_vehicles[0][0] == 0):
                control.throttle = 0.75
                control.brake = 0

        return control

    def save(self, tick_data):
        frame = self.step // 10

        # Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

        # Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net
        torch.cuda.empty_cache()

    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp_route):
        """
		Set the plan (route) for the agent
			used at leaderboard--> route scenario

		global_plan_gps:
		global_plan_world_coord: route -> ([0] -> waypoint.transform, [1] -> RoadOption)
		wp_route:  [0] -> waypoint, [1] -> RoadOption
		"""
        # waypoints for a road of the map
        self._global_route = wp_route
        # same
        # print("length of the global_plan_world_coord")
        # print(len(global_plan_world_coord))
        # print("length of the wp_route")
        # print(len(wp_route))

        # 50 means Maximum distance between samples
        # ds_ids record the key waypoints (every 50 waypoints or state change waypoints)
        ds_ids = downsample_route(global_plan_world_coord, 16)
        # print("length of the ds_ids")

        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        # print(len(self._global_plan))

        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]

    def _truncate_global_route_till_local_target(self, windows_size=5):
        ev_location = self._ego_vehicle.get_location()
        closest_idx = 0
        for i in range(len(self._global_route) - 1):
            if i > windows_size:
                break

            loc0 = self._global_route[i][0].transform.location
            loc1 = self._global_route[i + 1][0].transform.location

            wp_dir = loc1 - loc0
            wp_veh = ev_location - loc0
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                closest_idx = i + 1
        if closest_idx > 0:
            self._last_route_location = carla.Location(
                self._global_route[0][0].transform.location)

        self._global_route = self._global_route[closest_idx:]

    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
		Draw a list of waypoints at a certain height given in vertical_shift.
		"""
        for w in waypoints:
            wp = w[0].transform.location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].transform.location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].transform.location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)


def flatten(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


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
            distance_to_ego = actor['distance_to_ego']
            direction_to_ego = actor['direction_to_ego']
            speed = actor['speed']
            forward_vector = actor['forward_vector']
            training_list.append([distance_to_ego, direction_to_ego, speed, forward_vector, direction])
    training_list = training_list + [[0, 0, 0, 0, 0]] * (3 - len(training_list))
    return training_list
