from collections import OrderedDict
from dictor import dictor

import copy

from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration


from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict, create_default_json_msg, save_dict


class RouteIndexer():
    def __init__(self, routes_file, scenarios_file, repetitions):
        #
        #ROUTES=leaderboard/data/TCP_training_routes/routes_town01.xml
        #SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
        #
        self._routes_file = routes_file
        self._scenarios_file = scenarios_file
        self._repetitions = repetitions
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.routes_length = []
        self._index = 0

        # retrieve routes
        # new_config = RouteScenarioConfiguration()
        # from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
        # from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration

        route_configurations = RouteParser.parse_routes_file(self._routes_file, self._scenarios_file, False)

        self.n_routes = len(route_configurations)
        self.total = self.n_routes * self._repetitions
        # print("nums of config:")
        # print(self.n_routes)
        # 300
        for i, config in enumerate(route_configurations):
            # print("config")
            # print(config)
            # <srunner.scenarioconfigs.route_scenario_configuration.RouteScenarioConfiguration object>
            # every config means a route's start point and end point

            for repetition in range(repetitions):
                config.index = i * self._repetitions + repetition
                config.repetition_index = repetition
                self._configs_dict['{}.{}'.format(config.name, repetition)] = copy.copy(config)
                # print(config.name) # RouteScenario_295
                # print(config.trajectory) # [<carla.libcarla.Location object at 0x7f7769281f70>, <carla.libcarla.Location object at 0x7f7769281fb0>]
                # print(config.scenario_file) # leaderboard/data/scenarios/all_towns_traffic_scenarios.json
        # print(len(list(self._configs_dict.items())))
        # 300routes (you can find it at leaderboard/data/TCP_training_routes/routes_town01.xml)
        self._configs_list = list(self._configs_dict.items())

    def peek(self):
        """
        there is no new waypoints
        """
        return not (self._index >= len(self._configs_list))

    def next(self):
        '''
        every config means a route's start point and end point
        '''
        if self._index >= len(self._configs_list):
            return None

        key, config = self._configs_list[self._index]
        self._index += 1

        return config

    def resume(self, endpoint):
        data = fetch_dict(endpoint)

        if data:
            checkpoint_dict = dictor(data, '_checkpoint')
            if checkpoint_dict and 'progress' in checkpoint_dict:
                progress = checkpoint_dict['progress']
                if not progress:
                    current_route = 0
                else:
                    current_route, total_routes = progress
                if current_route <= self.total:
                    self._index = current_route
                else:
                    print('Problem reading checkpoint. Route id {} '
                          'larger than maximum number of routes {}'.format(current_route, self.total))

    def save_state(self, endpoint):
        data = fetch_dict(endpoint)
        if not data:
            data = create_default_json_msg()
        data['_checkpoint']['progress'] = [self._index, self.total]

        save_dict(endpoint, data)
