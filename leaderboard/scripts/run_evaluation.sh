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


# TCP evaluation
export ROUTES=leaderboard/data/evaluation_routes/test2.xml
#export ROUTES=leaderboard/data/TCP_training_routes/routes_town05_val.xml
export TEAM_AGENT=team_code/our_agent.py
export TEAM_CONFIG=/home/wyz/TCP/log/TCP_T/epoch=9-last.ckpt
export CHECKPOINT_ENDPOINT=t1.json
export SCENARIOS=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
export SAVE_PATH=data/t1/


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
--trafficManagerPort=${TM_PORT}


#export TEAM_CONFIG=/home/wyz/TCP/log/TCP_T_aggressive/epoch=9-last.ckpt
#export CHECKPOINT_ENDPOINT=results_TransformerV3_aggressive.json
#export SAVE_PATH=data/results_TransformerV3_aggressive/
#python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
#--scenarios=${SCENARIOS}  \
#--routes=${ROUTES} \
#--repetitions=${REPETITIONS} \
#--track=${CHALLENGE_TRACK_CODENAME} \
#--checkpoint=${CHECKPOINT_ENDPOINT} \
#--agent=${TEAM_AGENT} \
#--agent-config=${TEAM_CONFIG} \
#--debug=${DEBUG_CHALLENGE} \
#--record=${RECORD_PATH} \
#--resume=${RESUME} \
#--port=${PORT} \
#--trafficManagerPort=${TM_PORT}

#export TEAM_CONFIG=/home/wyz/TCP/log/TCP_T_cautious/epoch=9-last.ckpt
#export CHECKPOINT_ENDPOINT=results_TransformerV3_cautious.json
#export SAVE_PATH=data/results_TransformerV3_cautious/
#python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
#--scenarios=${SCENARIOS}  \
#--routes=${ROUTES} \
#--repetitions=${REPETITIONS} \
#--track=${CHALLENGE_TRACK_CODENAME} \
#--checkpoint=${CHECKPOINT_ENDPOINT} \
#--agent=${TEAM_AGENT} \
#--agent-config=${TEAM_CONFIG} \
#--debug=${DEBUG_CHALLENGE} \
#--record=${RECORD_PATH} \
#--resume=${RESUME} \
#--port=${PORT} \
#--trafficManagerPort=${TM_PORT}


