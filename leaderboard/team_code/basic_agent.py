import os
import json
import datetime
import pathlib
import time
import cv2

import torch
import carla
import numpy as np
import pickle
from PIL import Image

from leaderboard.autoagents import autonomous_agent
import numpy as np
from omegaconf import OmegaConf

from roach.criteria import run_stop_sign
# from roach.obs_manager.birdview.chauffeurnet import ObsManager
from roach.obs_manager.birdview.chauffeurnet_noimg import ObsManager
from roach.utils.config_utils import load_entry_point
import roach.utils.transforms as trans_utils
from roach.utils.traffic_light import TrafficLightHandler

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import downsample_route
from agents.navigation.local_planner import RoadOption

from team_code.planner import RoutePlanner

import traceback

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'ROACHAgent'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def get_xyz(_):
    return [_.x, _.y, _.z]


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1


TRAFFIC_LIGHT_STATE = {
    'None': 0,
    'Red': 1,
    'Green': 2,
    'Yellow': 3,
}


class ROACHAgent(autonomous_agent.AutonomousAgent):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)

    # def __call__() ---> self.run_step()
    # follows the rules fo inheritance and overriding
    # it will access its own corresponding member varible, it will access its own member variable

    # path_to_conf_file --> roach/config/config_agent.yaml

    def setup(self, path_to_conf_file, ckpt="roach/log/ckpt_11833344.pth"):
        self.behavior = 'cautious'
        self._render_dict = None
        self.supervision_dict = None
        self._ckpt = ckpt
        cfg = OmegaConf.load(path_to_conf_file)
        cfg = OmegaConf.to_container(cfg)
        self.cfg = cfg
        # config_agent.ymal
        self._obs_configs = cfg['obs_configs']
        self._train_cfg = cfg['training']

        #   entry_point: roach.models.ppo_policy:PpoPolicy
        #
        self._policy_class = load_entry_point(cfg['policy']['entry_point'])
        # PpoPolicy
        self._policy_kwargs = cfg['policy']['kwargs']
        # roach/config/config_agent.yaml

        if self._ckpt is None:
            self._policy = None
        else:
            self._policy, self._train_cfg['kwargs'] = self._policy_class.load(
                self._ckpt)
            self._policy = self._policy.eval()

        # roach.untils.rl_birdview_wrapper:RlBirdviewWrapper
        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']

        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()

        self.initialized = False

        self._3d_bb_distance = 50

        self.prev_lidar = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' %
                                             x, (now.month, now.day, now.hour, now.minute, now.second)))

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'measurements').mkdir()
            (self.save_path / 'supervision').mkdir()
            (self.save_path / 'bev').mkdir()

            (self.save_path / 'surroundings').mkdir()

    def _init(self):
        # RouteScenario _update_route -> interpolate_trajectory -> agent.set_global_plan
        # self._plan_gps_HACK have more waypoints to follow (eg: 139)
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)  # get self._global_plan

        # self._global_plan reduce the number of waypoints
        # min_distance, max_distance, debug_size(image?)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()
        self._ego_vehicle = CarlaDataProvider.get_ego()
        self._last_route_location = self._ego_vehicle.get_location()

        self._ego_vehicle_location = self._ego_vehicle.get_location()

        self._criteria_stop = run_stop_sign.RunStopSign(self._world)

        self.birdview_obs_manager = ObsManager(
            self.cfg['obs_configs']['birdview'], self._criteria_stop)
        self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

        self.navigation_idx = -1

        # for stop signs
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False  # if the ego vehicle is influenced by a stop sign

        TrafficLightHandler.reset(self._world)
        print("initialized")

        self.initialized = True

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

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

    def _get_position(self, tick_data):
        """
        gps to location
        return: gps
        """
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    # agent.set_global_plan
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
        ds_ids = downsample_route(global_plan_world_coord, 50)
        # print("length of the ds_ids")
        # print(len(ds_ids))
        self._global_plan = [global_plan_gps[x] for x in ds_ids]

        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]

        self._plan_gps_HACK = global_plan_gps
        self._plan_HACK = global_plan_world_coord

    def sensors(self):
        """
        autoagents/agent_wrapper --> setup_sensors

        """
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': -1.5, 'y': 0.0, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 900, 'height': 256, 'fov': 100,
                'id': 'rgb'
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

    def tick(self, input_data, timestamp):
        """
        input_data: autonomous_agent.sensor_interface.get_data() --> SensorInterface(Leaderboard.envs.xxx)
        return:
        tick_data

        """
        # print(input_data.keys())
        # dict_keys(['gps', 'imu', 'rgb', 'speed'])

        # define self._last_route_location
        # define self._globa_route
        self._truncate_global_route_till_local_target()
        # self._draw_waypoints(CarlaDataProvider.get_world(), self._global_route, vertical_shift=1.0, persistency=3)

        # birdview_obss_manager ---> roach.obs_manager.birdview.chauffeurnet  import ObsManager
        # return obs_dict = {'rendered': image, 'masks': masks} ---> rendered(test_save_path)
        birdview_obs = self.birdview_obs_manager.get_observation(self._global_route)

        surroundings_info = self.birdview_obs_manager.get_surroundings()

        # carladataprovider.get_ego().get_control()  --->
        # the client returns the control applied in the last tick
        # Return carla.VehicleControl
        control = self._ego_vehicle.get_control()
        # print(control)
        # VehicleControl(throttle=0.000000, steer=0.000000, brake=0.000000, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)

        throttle = np.array([control.throttle], dtype=np.float32)
        steer = np.array([control.steer], dtype=np.float32)
        brake = np.array([control.brake], dtype=np.float32)
        gear = np.array([control.gear], dtype=np.float32)

        ev_transform = self._ego_vehicle.get_transform()
        vel_w = self._ego_vehicle.get_velocity()
        vel_ev = trans_utils.vec_global_to_ref(vel_w, ev_transform.rotation)
        vel_xy = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)

        # run_stop_sign.RunStopSign(self._world)
        # roach/criteria/run_stop_sign
        # stop_sign_info = self._criteria_stop.tick(self._ego_vehicle, timestamp)
        self._criteria_stop.tick(self._ego_vehicle, timestamp)
        # print("stop_sign_info")
        # print(stop_sign_info)

        state_list = []
        state_list.append(throttle)
        state_list.append(steer)
        state_list.append(brake)
        state_list.append(gear)
        state_list.append(vel_xy)
        state = np.concatenate(state_list)
        # we cannot do this cause the other information may cause the fatal error
        # the surrounding information should be saved directly in "save" function
        obs_dict = {
            'state': state.astype(np.float32),
            # 'surrounding': surroundings_info,
            'birdview': birdview_obs['masks'],
        }

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        # how to change rgb to the surrounding information

        # 'gps': (7226, array([-0.00365808, -0.0033355 ,  0.0327018 ]))
        gps = input_data['gps'][1][:2]
        # 'speed': (7226, {'speed': 1.2146504072466647e-06})
        speed = input_data['speed'][1]['speed']
        # 'imu': (7226, array([-1.12413742e-04,  1.32158236e-03,  9.70741940e+00, -1.18394056e-03,
        # -5.58337662e-04,  1.16674928e-04,  5.01234198e+00]))
        compass = input_data['imu'][1][-1]

        target_gps, target_command = self.get_target_gps(input_data['gps'][1], compass)

        weather = self._weather_to_dict(self._world.get_weather())

        result = {
            'rgb': rgb,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'weather': weather,
        }

        # self._route_planner = RoutePlanner(4.0, 50.0)
        # self._route_planner.set_route(self._global_plan, True)
        next_wp, next_cmd = self._route_planner.run_step(self._get_position(result))

        result['next_command'] = next_cmd.value
        result['x_target'] = next_wp[0]
        result['y_target'] = next_wp[1]

        # print(result.keys())
        # dict_keys(['rgb', 'gps', 'speed', 'compass', 'weather', 'next_command', 'x_target', 'y_target'])

        # tick_data, policy_input, rendered, target_gps, target_command = self.tick(input_data, timestamp)
        return result, obs_dict, birdview_obs['rendered'], target_gps, target_command, surroundings_info

    # obs_dict --> obs_dict = {'state': state.astype(np.float32),'birdview': birdview_obs['masks'],}

    def im_render(self, render_dict):
        im_birdview = render_dict['rendered']
        h, w, c = im_birdview.shape
        im = np.zeros([h, w * 2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        action_str = np.array2string(render_dict['action'], precision=2, separator=',', suppress_small=True)

        txt_1 = f'a{action_str}'
        # put text on image
        im = cv2.putText(im, txt_1, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        debug_texts = [
            'should_brake: ' + render_dict['should_brake'],
        ]
        for i, txt in enumerate(debug_texts):
            im = cv2.putText(im, txt, (w, (i + 2) * 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return im

    @torch.no_grad()
    # redefine autonomous_agent's run_step()
    def run_step(self, input_data, timestamp):
        # traceback.print_stack()
        # input_data: gps, imu, rgb, speed
        if not self.initialized:
            self._init()

        self.step += 1

        if self.step < 20:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            self.last_control = control
            return control

        if self.step % 2 != 0:
            return self.last_control

        #  rendered --> image?

        #  return result, obs_dict, birdview_obs['rendered'], target_gps, target_command
        tick_data, policy_input, rendered, target_gps, target_command, surroundings_info = self.tick(input_data,
                                                                                                     timestamp)

        # what is the policy_input(obs_dict)
        # obs_dict = {'state': state.astype(np.float32),'birdview': birdview_obs['masks'],}

        gps = self._get_position(tick_data)

        # The program will find the farthest point within the min_distance range

        # Since the _waypoint_planner includes all waypoints
        # so its farthest waypoints means near waypoints
        # 4, 50 more waypoints(100+) RoutePlanner(4.0, 50)
        near_node, near_command = self._waypoint_planner.run_step(gps)

        # 7.5, 25 less waypoints(10+-) RoutePlanner(7.5, 25.0, 257)
        far_node, far_command = self._command_planner.run_step(gps)
        # node should be waypoint np.array([pos['lat'], pos['lon']])
        # command should be RoadOption

        # use config's policy to finish the cruise
        # we need to change the policy_input's image
        actions, values, log_probs, mu, sigma, features = self._policy.forward(
            policy_input, deterministic=True, clip_action=True)
        if self.behavior == 'aggressive':
            mu[0][0] = mu[0][0] * 1.15
            sigma[0][0] = sigma[0][0] * 0.85
        elif self.behavior == 'cautious':
            mu[0][0] = mu[0][0] * 0.85
            sigma[0][0] = sigma[0][0] * 1.15

        def _get_action_beta(alpha1, beta1):
            alpha = alpha1.reshape(1, 2)
            beta = beta1.reshape(1, 2)
            x = np.zeros_like(alpha)
            x[:, 1] += 0.5
            mask1 = (alpha > 1) & (beta > 1)
            x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)

            mask2 = (alpha <= 1) & (beta > 1)
            x[mask2] = 0.0

            mask3 = (alpha > 1) & (beta <= 1)
            x[mask3] = 1.0

            # mean
            mask4 = (alpha <= 1) & (beta <= 1)

            temp = (alpha[mask4] + beta[mask4])
            x[mask4] = alpha[mask4] / temp.clip(1e-5)

            x = x * 2 - 1

            return -x
        # print(actions, sigma, mu, _get_action_beta(sigma, mu))
        actions = _get_action_beta(sigma, mu)


        # take actions to control ---> VehicleControl
        control = self.process_act(actions)

        render_dict = {"rendered": rendered, "action": actions}

        # additional collision detection to enhance safety
        should_brake = self.collision_detect()
        only_ap_brake = True if (control.brake <= 0 and should_brake) else False
        if should_brake:
            control.steer = control.steer * 0.5
            control.throttle = 0.0
            control.brake = 1.0
        render_dict = {"rendered": rendered, "action": actions, "should_brake": str(should_brake), }

        # bev image(bird's eye view fused with Multiple Sensor Data)
        # only save the image at town01(we need to adjust the project)
        render_img = self.im_render(render_dict)
        # render_img = [] # donnot save img

        supervision_dict = {
            'action': np.array([control.throttle, control.steer, control.brake], dtype=np.float32),
            'value': values[0],
            'action_mu': mu[0],
            'action_sigma': sigma[0],
            'features': features[0],
            'speed': tick_data['speed'],
            'target_gps': target_gps,
            'target_command': target_command,
            'should_brake': should_brake,
            'only_ap_brake': only_ap_brake,
        }

        ego_vehicle_waypoint = self._map.get_waypoint(self._ego_vehicle.get_location())
        # print(ego_vehicle_waypoint.get_left_lane())
        # print(ego_vehicle_waypoint.get_right_lane())
        traffic_light_state, light_loc, light_id = TrafficLightHandler.get_light_state(self._ego_vehicle)

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

        # 	print(wp_end)
        #
        # print(len(self._global_route))

        surroundings_dict = {
            'is_junction': 1 if ego_vehicle_waypoint.is_junction else 0,
            # 'is_at_traffic_light': self._ego_vehicle.is_at_traffic_light(),
            'traffic_light_state': TRAFFIC_LIGHT_STATE[str(traffic_light_state)],  # 0, 1, 2, 3 None, Red, Green, Yellow
            'wp': wp_end,

            # 'traffic_sign':1,
            # 'waypoints':1,
            # 'intersection': True, # False
            # 'left_vehicles':1,
            # 'right_vehicles':1,
            # 'front_vehicles':1,
            # 'intersection_left_vehicles':1,
            # 'intersection_right_vehicles':1,
            # 'intersection_front_vehicles':1, # use for unprotection leftturn
            # 'intersection_left_pedestrians':1,
            # 'intersection_right_pedestrians':1,
            # 'lane_invasion_event':1, #left,right
            # 'front_brake_light_status':True, #False
            # 'left_vehicle_right_signal_status':True,
            # 'right_vehicle_left_signal_status':True,
        }
        surroundings_dict.update(surroundings_info)

        if SAVE_PATH is not None and self.step % 10 == 0:
            # save surroundings
            self.save(near_node, far_node, near_command, far_command, tick_data,
                      supervision_dict, render_img, should_brake, surroundings_dict)

        steer = control.steer
        control.steer = steer + 1e-2 * np.random.randn()
        self.last_control = control
        return control

    def collision_detect(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        self.is_vehicle_present = 1 if vehicle is not None else 0
        self.is_pedestrian_present = 1 if walker is not None else 0

        return any(x is not None for x in [vehicle, walker])

    def _is_walker_hazard(self, walkers_list):
        z = self._ego_vehicle.get_location().z
        p1 = _numpy(self._ego_vehicle.get_location())
        v1 = 10.0 * _orientation(self._ego_vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._ego_vehicle.get_location().z

        o1 = _orientation(self._ego_vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._ego_vehicle.get_location())
        s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._ego_vehicle.get_velocity())))  # increases the threshold distance
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._ego_vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None

    def save(self, near_node, far_node, near_command, far_command, tick_data, supervision_dict, render_img,
             should_brake, surroundings_dict):
        frame = self.step // 10 - 2

        # save image and bev
        # Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        # Image.fromarray(render_img).save(self.save_path / 'bev' / ('%04d.png' % frame))

        pos = self._get_position(tick_data)
        # tick_data['gps']
        # [0.00168161 0.00313114]
        # pos
        # [187.20484611 348.55691576]

        theta = tick_data['compass']
        speed = tick_data['speed']

        data = {
            'x': pos[0],
            'y': pos[1],
            'theta': theta,
            'speed': speed,
            'x_command_far': far_node[0],
            'y_command_far': far_node[1],
            'command_far': far_command.value,  # RoadOption.value
            'x_command_near': near_node[0],
            'y_command_near': near_node[1],
            'command_near': near_command.value,
            'should_brake': should_brake,
            'x_target': tick_data['x_target'],
            'y_target': tick_data['y_target'],
            'target_command': tick_data['next_command'],
        }
        outfile = open(self.save_path / 'measurements' / ('%04d.json' % frame), 'w')
        json.dump(data, outfile, indent=4)
        outfile.close()

        with open(self.save_path / 'supervision' / ('%04d.npy' % frame), 'wb') as f:
            np.save(f, supervision_dict)

        # print("roach_ap_noimng_agent")
        # print(surroundings_dict)
        with open(self.save_path / 'surroundings' / ('%04d.npy' % frame), 'wb') as f:
            np.save(f, surroundings_dict)
        # with open(self.save_path / 'surroundings' / ('%04d.pkl' % frame), 'wb') as f:

    # 	pickle.dump(surroundings_dict, f)

    def get_target_gps(self, gps, compass):
        # target gps
        def gps_to_location(gps):
            lat, lon, z = gps
            lat = float(lat)
            lon = float(lon)
            z = float(z)

            location = carla.Location(z=z)
            xy = (gps[:2] - self._command_planner.mean) * self._command_planner.scale
            location.x = xy[0]
            location.y = -xy[1]
            return location

        global_plan_gps = self._global_plan
        next_gps, _ = global_plan_gps[self.navigation_idx + 1]
        next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
        next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass) - 90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(next_vec_in_global, ref_rot_in_global)

        if np.sqrt(loc_in_ev.x ** 2 + loc_in_ev.y ** 2) < 12.0 and loc_in_ev.x < 0.0:
            self.navigation_idx += 1

        self.navigation_idx = min(self.navigation_idx, len(global_plan_gps) - 2)

        _, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
        gps_point, road_option_1 = global_plan_gps[self.navigation_idx + 1]
        gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])

        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)

    def process_act(self, action):

        # acc, steer = action.astype(np.float64)
        acc = action[0][0]
        steer = action[0][1]
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control

    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }

        return weather

    def _get_3d_bbs(self, max_distance=50):

        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "pedestrians": []
        }

        bounding_boxes['traffic_lights'] = self._find_obstacle_3dbb('*traffic_light*', max_distance)
        bounding_boxes['stop_signs'] = self._find_obstacle_3dbb('*stop*', max_distance)
        bounding_boxes['vehicles'] = self._find_obstacle_3dbb('*vehicle*', max_distance)
        bounding_boxes['pedestrians'] = self._find_obstacle_3dbb('*walker*', max_distance)

        return bounding_boxes

    def _find_obstacle_3dbb(self, obstacle_type, max_distance=50):
        """Returns a list of 3d bounding boxes of type obstacle_type.
        If the object does have a bounding box, this is returned. Otherwise a bb
        of size 0.5,0.5,2 is returned at the origin of the object.

        Args:
            obstacle_type (String): Regular expression
            max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

        Returns:
            List: List of Boundingboxes
        """
        obst = list()

        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(self._ego_vehicle.get_location())

            if 0 < distance_to_car <= max_distance:

                if hasattr(_obstacle, 'bounding_box'):
                    loc = _obstacle.bounding_box.location
                    _obstacle.get_transform().transform(loc)

                    extent = _obstacle.bounding_box.extent
                    _rotation_matrix = self.get_matrix(
                        carla.Transform(carla.Location(0, 0, 0), _obstacle.get_transform().rotation))

                    rotated_extent = np.squeeze(
                        np.array((np.array([[extent.x, extent.y, extent.z, 1]]) @ _rotation_matrix)[:3]))

                    bb = np.array([
                        [loc.x, loc.y, loc.z],
                        [rotated_extent[0], rotated_extent[1], rotated_extent[2]]
                    ])

                else:
                    loc = _obstacle.get_transform().location
                    bb = np.array([
                        [loc.x, loc.y, loc.z],
                        [0.5, 0.5, 2]
                    ])

                obst.append(bb)

        return obst

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

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