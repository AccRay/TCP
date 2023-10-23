TCP/
	data.py
		修改了训练时读入的数据，改成了读入我们采集的数据，而不是图像
	config.py
		删除了轨迹部分的参数
	eval.py
		绘制注意力图
	model.eval.py
		将model_transformerV3.py改为评测模式绘制注意力图
	model_old.py
		TCP没有轨迹
	model_TCP.py
		完美场景下的TCP模型
	model_transformerV3.py
		我们的方法
	train.py
		修改了计算loss的方式，删除掉了轨迹和速度部分
	train_style.py
		风格版本finetune
		
tools/
	filiter_style_data.py
		风格版本
	filter_data.py
		增加了Waypoint以及其他我们采集的数据的数据预处理
	gen_data.py
		增加了Waypoint以及其他我们采集的数据的数据预处理
	gen_style_data.py
		风格版本
		
roach/obs_manager_bridvier
	tcp_noming.py
		用于我们的模型获得完美场景下的数据
		
leaderboard/
	scripts/
		style_collection.sh
			采集风格数据
	leaderboard/utils
		statistics_manager.py
			增加了我们的评价指标
	data/
		增加了新的评测路线
	team_code/
		basic_agent.py
			风格Roach用于采集数据
		our_agent.py
			我们的智能体，改了导航模式，输出动作方式，采集的数据等，加了评价指标
		roach_ap_noimg_agent_old.py
			没有评价指标的Roach，用于产生数据
		roach_ap_noimg_agent.py
			带评价指标，用于评价Roach
			
			
		
	
