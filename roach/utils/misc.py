import math,re
import carla
import numpy as np
from enum import Enum

# https://carla.readthedocs.io/en/latest/python_api/#carla.LandmarkType
HELPER_LAND_MARK = {
    "Danger": 101,
    "LanesMerging": 121,
    "CautionPedestrian": 133,
    "CautionBicycle": 138,
    "LevelCrossing": 150,
    "StopSign": 206,
    "YieldSign": 205,
    "MandatoryTurnDirection": 209,
    "MandatoryLeftRightDirection": 211,
    "TwoChoiceTurnDirection": 214,
    "Roundabout": 215,
    "PassRightLeft": 222,
    "AccessForbidden": 250,
    "AccessForbiddenMotorvehicles": 251,
    "AccessForbiddenTrucks": 253,
    "AccessForbiddenBicycle": 254,
    "AccessForbiddenWeight": 263,
    "AccessForbiddenWidth": 264,
    "AccessForbiddenHeight": 265,
    "AccessForbiddenWrongDirection": 267,
    "ForbiddenUTurn": 272,
    "MaximumSpeed": 274,
    "ForbiddenOvertakingMotorvehicles": 276,
    "ForbiddenOvertakingTrucks": 277,
    "AbsoluteNoStop": 283,
    "RestrictedStop": 286,
    "HasWayNextIntersection": 301,
    "PriorityWay": 306,
    "PriorityWayEnd": 307,
    "CityBegin": 310,
    "CityEnd": 311,
    "Highway": 330,
    "RecomendedSpeed": 380,
    "RecomendedSpeedEnd": 381
}


def get_helper_landmarks(world=None, waypoint=None, distance=50.0, getAll=False):
    if getAll:
        current_map = world.get_map()
        landmarks = current_map.get_all_landmarks()
    else:
        landmarks = waypoint.get_landmarks(distance, True)

    # the actor's stopsign is not the landmark stopsigns
    
    record_land_mark ={
        # "MaximumSpeed": None,
        "MaximumSpeed": 0,
        "StopSign": 0,
        "YieldSign": 0,
    }
    for landmark in landmarks:
        landmark_type = int(landmark.type)
        if landmark_type in HELPER_LAND_MARK.values():
            # landmark_name = [key for key, value in HELPER_LAND_MARK.items() if value == 274]
            if landmark_type == 274:
                match = re.search(r"(?i)(Speed|speed)_(\d+)", landmark.name)
                speed_limit = match.group(2)
                record_land_mark["MaximumSpeed"] = speed_limit
                # print(speed_limit)
            elif landmark_type == 206:
                if str(landmark.orientation) == 'Negative':
                    continue
                # stop_sign = True
                # record_land_mark["StopSign"] = stop_sign
                record_land_mark["StopSign"] = 1
                # print("Landmark Orientation", landmark.orientation)
                # print([landmark.transform.location.x, landmark.transform.location.y, landmark.transform.location.z])
                # add orientation
                # print("stop_sign:", stop_sign)
            elif landmark_type == 205:
                yield_sign = True
                record_land_mark["YieldSign"] = 1
                # print("yield_sign:", yield_sign)
            else:
                print("misc.py")
                print("Landmark ID:", landmark.id)
                print("Landmark NAME:", landmark.name)
                print("Landmark Type:", landmark.type)
                print("Landmark Location:", landmark.transform.location)
                print("Landmark Rotation:", landmark.transform.rotation)
                print("-"*20)
            # print("record_land_mark")
            # print(record_land_mark)
            # print("-" * 20)
    return record_land_mark

def calculate_angle_between_front_and_current(current_location, current_rocation, front_location):
    """
    return: 
    left:True, right:False
    angle_between_actors: return degrees
    """
    dx = front_location.x - current_location.x
    dy = front_location.y - current_location.y
    ev_fv_vector = [dx, dy, 0]

    ev_yaw_rad = math.radians(current_rocation.yaw)
    ev_vector = [math.cos(ev_yaw_rad), math.sin(ev_yaw_rad),0]

    unit_ev_fv_vector = ev_fv_vector / np.linalg.norm(ev_fv_vector)
    unit_ev_vector    = ev_vector / np.linalg.norm(ev_vector)
    dot_product       = np.dot(unit_ev_fv_vector, unit_ev_vector)
    angle_radians     = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_between_actors     = np.degrees(angle_radians)


    cross_product = np.cross(ev_fv_vector, ev_vector)
    is_left = cross_product[2] > 0

    return is_left, angle_between_actors

def calculate_direction_angle_between_front_and_current(current_rocation, front_rotation):

    fv_yaw_rad = math.radians(front_rotation.yaw)
    fv_vector = [math.cos(fv_yaw_rad), math.sin(fv_yaw_rad),0]

    ev_yaw_rad = math.radians(current_rocation.yaw)
    ev_vector = [math.cos(ev_yaw_rad), math.sin(ev_yaw_rad),0]



    unit_fv_vector = fv_vector / np.linalg.norm(fv_vector)
    unit_ev_vector = ev_vector / np.linalg.norm(ev_vector)
    dot_product    = np.dot(unit_fv_vector, unit_ev_vector)
    angle_radians  = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_between_actors_direction = np.degrees(angle_radians)

    return angle_between_actors_direction

def calculate_speed_magnitue(actor):
    '''
    m/s
    '''
    velocity = actor.get_velocity()
    speed = velocity.x**2 + velocity.y**2 + velocity.z**2
    speed = speed ** 0.5
    return speed

def calculate_acceleration_magnitue(actor):
    '''
    m/s2
    '''
    acceleration = actor.get_acceleration()
    acc = acceleration.x**2 + acceleration.y**2 + acceleration.z**2
    acc = acc ** 0.5
    return acc

def get_lane_center(map, location):
    """Project current loction to its lane center, return lane center waypoint"""

    lane_center = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving|carla.LaneType.Shoulder|carla.LaneType.Sidewalk)
    if lane_center.is_junction:
        shoulder = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType)

    return lane_center


def is_within_distance_ahead(target_location, current_location, current_transform, max_distance):
    
    """
      Check if a target object is within a certain distance in front of a reference object.

      :param target_location: location of the target object
      :param current_location: location of the reference object
      :param current_transform: transform of the reference object
      :param max_distance: maximum allowed distance
      :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)
    if norm_target < 0.001:
        return True
    if norm_target > max_distance:
        return False

    # forward_vector = np.array(
    #     [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(
        np.clip(np.dot(forward_vector, target_vector) / norm_target, -1, 1)))

    return 0.0 < d_angle < 90.0

def calculate_distance_between_location(location_A, location_B):
    return location_A.distance(location_B)

def is_vehicle_on_road(world, vehicle_bbox):
    location = vehicle_bbox[1]
    waypoint = world.get_map().get_waypoint(location, project_to_road=True)
    return waypoint is not None
    # for actor in collision_actor:
    #     if 'road' in actor.type_id:
    #         return True
    # return False
