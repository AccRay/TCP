from collections import deque
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.layernorm1 = nn.LayerNorm([9, 64])
		self.layernorm2 = nn.LayerNorm([9, 64])
		self.MLP = nn.Sequential(
			nn.Linear(64, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64)
		)
		self.multihead_attn = nn.MultiheadAttention(64, 4, batch_first=True)
		
	def forward(self, X):
		attn_output, attn_output_weights = self.multihead_attn(X, X, X)
		attn_output = attn_output + X
		att_feature = self.layernorm1(attn_output)
		feature_emb = self.layernorm2(self.MLP(att_feature) + att_feature)
		return attn_output_weights, feature_emb


class TCP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config

		# using sequential to expressing a neutral network
		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)

		self.join_ctrl = nn.Sequential(
							nn.Linear(128+576, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)
		# shared branches_neurons
		dim_out = 2

		self.policy_head = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)
		self.output_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
			)
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 576),
				nn.Softmax(1)
			)

		self.wp_mlp = nn.Sequential(
			nn.Linear(32, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 256),
			nn.ReLU(inplace=True)
		)
		self.wp_att = nn.Sequential(
				nn.Linear(256+256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 576),
				nn.Softmax(1)
			)

		self.merge = nn.Sequential(
				nn.Linear(576+256, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
			)

		self.is_junction_emb = nn.Linear(1, 64, bias=False)
		self.vehicles_emb = nn.Linear(45, 64, bias=False)
		self.walkers_emb = nn.Linear(45, 64, bias=False)
		self.stops_emb = nn.Linear(3, 64, bias=False)
		self.max_speed_emb = nn.Linear(1, 64, bias=False)
		self.stop_sign_emb = nn.Linear(1, 64, bias=False)
		self.yield_sign_emb = nn.Linear(1, 64, bias=False)
		self.traffic_light_state = nn.Linear(4, 64, bias=False)
		self.waypoint_emb = nn.Linear(32, 64, bias=False)
		self.tf_layer1 = TransformerBlock()
		self.tf_layer2 = TransformerBlock()
		self.tf_layer3 = TransformerBlock()
		self.tf_layer4 = TransformerBlock()
		# self.env_feature = nn.Linear(576, 1024)
		self.env_feature2 = nn.Linear(576, 576)

	#nn caculate 
	def forward(self, is_junction, vehicles, walkers, stops, max_speed, stop_sign, yield_sign,
				traffic_light_state, state, target_point, waypoints):
		# feature_emb, cnn_feature = self.perception(img)
		is_junction_emb = self.is_junction_emb(is_junction)
		vehicles_emb = self.vehicles_emb(vehicles)
		walkers_emb = self.walkers_emb(walkers)
		stops_emb = self.stops_emb(stops)
		max_speed_emb = self.max_speed_emb(max_speed)
		stop_sign_emb = self.stop_sign_emb(stop_sign)
		yield_sign_emb = self.yield_sign_emb(yield_sign)
		traffic_light_state = self.traffic_light_state(traffic_light_state)
		waypoints_emb = self.waypoint_emb(waypoints)
		cat_emb_feature = torch.cat((is_junction_emb, vehicles_emb, walkers_emb, stops_emb, max_speed_emb,
									 stop_sign_emb, yield_sign_emb, traffic_light_state, waypoints_emb), dim=1).reshape(-1, 9, 64)
		_, feature_emb = self.tf_layer1(cat_emb_feature)
		_, feature_emb = self.tf_layer2(feature_emb)
		_, feature_emb = self.tf_layer3(feature_emb)
		_, feature_emb = self.tf_layer4(feature_emb)
		feature_emb = feature_emb.reshape(-1, 576)
		outputs = {}
		measurement_feature = self.measurements(state)

		traj_hidden_state = self.wp_mlp(waypoints)

		init_att = self.init_att(measurement_feature)
		feature_emb = self.env_feature2(F.relu(feature_emb))
		feature_emb = feature_emb * init_att
		j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
		outputs['pred_features_ctrl'] = j_ctrl
		policy = self.policy_head(j_ctrl)
		outputs['mu_branches'] = self.dist_mu(policy)
		outputs['sigma_branches'] = self.dist_sigma(policy)

		x = j_ctrl
		mu = outputs['mu_branches']
		sigma = outputs['sigma_branches']
		future_feature, future_mu, future_sigma = [], [], []

		# initial hidden variable to GRU
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for _ in range(self.config.pred_len):
			x_in = torch.cat([x, mu, sigma], dim=1)
			h = self.decoder_ctrl(x_in, h)
			wp_att = self.wp_att(torch.cat([h, traj_hidden_state], 1))
			new_feature_emb = feature_emb * wp_att
			merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
			dx = self.output_ctrl(merged_feature)
			x = dx + x

			policy = self.policy_head(x)
			mu = self.dist_mu(policy)
			sigma = self.dist_sigma(policy)
			future_feature.append(x)
			future_mu.append(mu)
			future_sigma.append(sigma)


		outputs['future_feature'] = future_feature
		outputs['future_mu'] = future_mu
		outputs['future_sigma'] = future_sigma
		return outputs

	def process_action(self, pred, command, speed, target_point):
		action = self._get_action_beta(pred['mu_branches'].view(1,2), pred['sigma_branches'].view(1,2))
		acc, steer = action.cpu().numpy()[0].astype(np.float64)
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)
		steer = np.clip(steer, -1, 1)
		brake = np.clip(brake, 0, 1)

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}
		return steer, throttle, brake, metadata

	def _get_action_beta(self, alpha, beta):
		x = torch.zeros_like(alpha)
		x[:, 1] += 0.5
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)

		x = x * 2 - 1

		return x

	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1,2), sigma.view(1,2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			# generize  00000
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, steer, brake