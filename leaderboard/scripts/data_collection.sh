#!/bin/bash

export CARLA_ROOT=/home/wyz/CARLA_0.9.10
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000
export TM_PORT=8000
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export DATA_COLLECTION=True


# Roach data collection


# export TEAM_AGENT=team_code/roach_ap_agent.py
export TEAM_AGENT=team_code/roach_ap_noimg_agent.py


export TEAM_CONFIG=roach/config/config_agent.yaml


export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json


export ROUTES=leaderboard/data/TCP_training_routes/routes_town01.xml

# export ROUTES=leaderboard/data/TCP_training_routes/routes_town10_val.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town07.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town07_val.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town07_addition.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town06.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town06_val.xml
# export ROUTES=leaderboard/data/TCP_training_routes/routes_town04_addition.xml

# export CHECKPOINT_ENDPOINT=data_collect_town02_val_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town01_addition_results.json
export CHECKPOINT_ENDPOINT=data_collect_town01_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town03_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town10_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town10_addition_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town01_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town03_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town10_results.json
export CHECKPOINT_ENDPOINT=data_collect_town10_addition_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town10_val_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town07_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town07_val_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town06_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town06_val_results.json
# export CHECKPOINT_ENDPOINT=data_collect_town04_addition_results.json

# export SAVE_PATH=data/data_collect_town02_results/
# export SAVE_PATH=data/town02_val/
# export SAVE_PATH=data/town01_addition/ 

export SAVE_PATH=data/town01/
#export SAVE_PATH=data/town01_val/
#export SAVE_PATH=data/town01_addition/
#export SAVE_PATH=data/town02/
#export SAVE_PATH=data/town02_val/
#export SAVE_PATH=data/town03/
#export SAVE_PATH=data/town03_val/
#export SAVE_PATH=data/town03_addition/
#export SAVE_PATH=data/town04/

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}\
