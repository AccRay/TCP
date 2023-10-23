import os
import json
import datetime
import pathlib
import time
import cv2
from collections import deque
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import numpy as np
from model_eval import TCP
from config import GlobalConfig
from TCP.data import CARLA_Data
from d2l import torch as d2l
torch.manual_seed(24)

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(8, 8),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_attention-cues`"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    plt.xticks([])
    plt.yticks([])
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.tick_params(labelsize=4)
    plt.savefig('./1.svg',facecolor='w', edgecolor='w', dpi=600, format='svg',
                transparent=True, bbox_inches='tight')


def flatten(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
TRAFFIC_LIGHT_STATE = {
    'None': 0,
    'Red': 1,
    'Green': 2,
    'Yellow': 3,
}
config = GlobalConfig()
# train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)

state = torch.zeros([1, 9]).to('cuda')
target_point = torch.zeros([1, 2]).to('cuda', dtype=torch.float32)

is_junction_temp = 1
# is_junction_temp = 0
vehicles_temp = [[31.452871322631836, 36.61920340528985, 1.798332075189843e-05, 40.49681091308593, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
walkers_temp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
stops_temp = [0, 0, 0]
max_speed_temp = 0
stop_sign_temp = [0]
yield_sign_temp = [0]
traffic_light_state_temp = 'Red'
wp_end = [[-0.00110626220703125, 0.000263214111328125, -0.037027664482593536], [-0.7527618408203125, 1.0028610229492188, -0.037027664482593536], [-1.3714218139648438, 2.092571258544922, -0.037027664482593536], [-1.8471221923828125, 3.251842498779297, -0.037027664482593536], [-2.1722183227539062, 4.462017059326172, -0.037027664482593536], [-2.341461181640625, 5.70361328125, -0.037027664482593536], [-2.366973876953125, 6.8472900390625, -0.037027664482593536], [-2.3668212890625, 7.8472900390625, -0.037027664482593536], [-2.3666534423828125, 8.947837829589844, -0.037027664482593536], [-2.3666534423828125, 8.947837829589844, -0.037027664482593536], [-2.3665008544921875, 9.947837829589844, -0.037027664482593536], [-2.3663482666015625, 10.947837829589844, -0.037027664482593536], [-2.3661956787109375, 11.947837829589844, -0.037027664482593536], [-2.3660430908203125, 12.947837829589844, -0.037027664482593536], [-2.3658905029296875, 13.947837829589844, -0.037027664482593536], [-2.3657379150390625, 14.947837829589844, -0.037027664482593536]]

net = TCP(config)
path_to_conf_file = '/home/wyz/TCP/log/Attention/epoch=59-last.ckpt'
# path_to_conf_file = '/home/wyz/TCP/log/TCP_T/epoch=9-last.ckpt'
ckpt = torch.load(path_to_conf_file, map_location='cuda:0')
new_state_dict = OrderedDict()
for key, value in ckpt.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
net.load_state_dict(new_state_dict, strict=False)
net.cuda()
net.eval()
is_junction = torch.tensor(is_junction_temp * 5, dtype=torch.float32).view(1, 1).to('cuda')
vehicles = torch.tensor(flatten(vehicles_temp), dtype=torch.float32).view(1, -1).to('cuda')
walkers = torch.tensor(flatten(walkers_temp), dtype=torch.float32).view(1, -1).to('cuda')
stops = torch.tensor(stops_temp, dtype=torch.float32).view(1, -1).to('cuda')
max_speed = torch.tensor(int(max_speed_temp), dtype=torch.float32).view(1, 1).to('cuda')
stop_sign = torch.tensor(stop_sign_temp, dtype=torch.float32).view(1, -1).to('cuda')
yield_sign = torch.tensor(yield_sign_temp, dtype=torch.float32).view(1, -1).to('cuda')
traffic_light_state = TRAFFIC_LIGHT_STATE[str(traffic_light_state_temp)]
if traffic_light_state not in [0, 1, 2, 3]:
        traffic_light_state = 0
assert traffic_light_state in [0, 1, 2, 3]
traffic_light_state_one_hot = [0] * 4

traffic_light_state_one_hot[traffic_light_state] = 1
# traffic_light_state_one_hot[traffic_light_state] = 1
traffic_light_state_end = torch.tensor(traffic_light_state_one_hot * 10, dtype=torch.float32).view(1, -1).to('cuda')
wp_temp = torch.tensor(wp_end, dtype=torch.float32)
waypoints_end = wp_temp[:, 0:2].reshape(-1).view(1, -1).to('cuda')


pred, att1, att2, att3, att4 = net(is_junction, vehicles, walkers, stops, max_speed, stop_sign, yield_sign,
                        traffic_light_state_end, state, target_point, waypoints_end)
att1 = att1.to('cpu')
att2 = att2.to('cpu')
att3 = att3.to('cpu')
att4 = att4.to('cpu')
att = (att1 + att2 + att3 + att4) / 4
# show_heatmaps(att.reshape(1, 1, 9, 9), xlabel="Keys", ylabel="Queries")
att_sum = att.sum(dim=1)
att_sum = att_sum / att_sum.sum()
show_heatmaps(att_sum.reshape(1, 1, 1, 9), xlabel="", ylabel="Queries")
