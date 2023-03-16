from agent.agent import Agent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import datetime
import imageio
import os
from tensorflow.keras.models import load_model


def buy_save(buy, dir_name, i):
	buy = buy.reshape([-1, 3])
	df = pd.DataFrame(buy, columns=["buy_price", "time", "num"])
	df.to_csv("{}/buy_sell/buy_{}.csv".format(dir_name, i), index=False)
	return None


def sell_save(sell, dir_name, i):
	sell = sell.reshape([-1, 3])
	df = pd.DataFrame(sell, columns=["sell_price", "time", "num"])
	df.to_csv("{}/buy_sell/sell_{}.csv".format(dir_name, i), index=False)
	return None


def total_save(total, dir_name):
	total = total.reshape([-1, 2])
	df = pd.DataFrame(total, columns=["total_profit", "episode"])
	df.to_csv("{}/total_reward/total_reward.csv".format(dir_name, index=False))
	return None


def make_folder(episode, setting):
	dir_name = '_e'+str(episode)+setting
	os.mkdir(dir_name)
	os.mkdir(dir_name + '/models')
	os.mkdir(dir_name + '/buy_sell')
	os.mkdir(dir_name + '/total_reward')
	os.mkdir(dir_name + '/buy_sell_graph')
	os.mkdir(dir_name + '/num')
	os.mkdir(dir_name + '/graph_result')
	os.mkdir(dir_name + '/eval')
	return dir_name


def check_time(t_sec, dir_name):
	t_result = datetime.timedelta(seconds=t_sec)
	fo = open("{}/time.txt".format(dir_name), "w")
	fo.write(str(t_result))
	fo.close()
	return None


def graph(data, dir_name, time_len, episode):
	total = pd.read_csv("{}/total_reward/total_reward.csv".format(dir_name))
	plt.figure(figsize=(16, 8))
	plt.plot(total["episode"], total["total_profit"])
	plt.xlabel("episode")
	plt.ylabel("return")
	plt.savefig("{}/total_reward/total_graph.png".format(dir_name))

	for i in range(episode):
		buy = pd.read_csv("{}/buy_sell/buy_{}.csv".format(dir_name,i))
		sell = pd.read_csv("{}/buy_sell/sell_{}.csv".format(dir_name,i))
		fig=plt.figure(figsize=(24, 8))
		plt.xlabel("Time")
		plt.ylabel("Price")
		plt.plot(data.index[:time_len], data["Price"][:time_len], color='k')
		plt.scatter(buy["time"], buy["buy_price"], color='r')
		plt.scatter(sell["time"], sell["sell_price"], color='b')
		plt.title("episode_{}".format(i))
		plt.savefig("{}/buy_sell_graph/episode_{}.png".format(dir_name, i))
		plt.close(fig)

	for i in range(episode):
		buy = pd.read_csv("{}/buy_sell/buy_{}.csv".format(dir_name, i))
		sell = pd.read_csv("{}/buy_sell/sell_{}.csv".format(dir_name, i))
		fig=plt.figure(figsize=(24, 8))
		plt.xlabel("Time")
		plt.ylabel("Num")
		plt.ylim(-100, 100)
		plt.bar(buy["time"], buy["num"], color='b')
		plt.bar(sell["time"], sell["num"], color='b')
		plt.title("episode_{}".format(i))
		plt.savefig("{}/num/num_episode_{}.png".format(dir_name, i))
		plt.close(fig)

	directory = r'{}/graph_result/'.format(dir_name)  # 저장할 폴더
	save_name = r'buy_sell_ani'  # 저장할 gif 파일 이름
	speed = {'duration': 0.1}
	images = []
	for i in range(0, episode, 10):
		images.append(imageio.imread("{}/buy_sell_graph/episode_{}.png".format(dir_name, i)))
	imageio.mimsave('{}/{}.gif'.format(directory, save_name), images, **speed)

	directory = r'{}/graph_result/'.format(dir_name)  # 저장할 폴더
	save_name = r'num_ani'  # 저장할 gif 파일 이름
	speed = {'duration': 0.1}
	images = []
	for i in range(0, episode, 10):
		images.append(imageio.imread("{}/num/num_episode_{}.png".format(dir_name, i)))
	imageio.mimsave('{}/{}.gif'.format(directory, save_name), images, **speed)
	return None


def eval_buy_save(buy, dir_name):
	buy = buy.reshape([-1, 3])
	df = pd.DataFrame(buy, columns=["buy_price", "time", "num"])
	df.to_csv("{}/eval/buy.csv".format(dir_name), index=False)
	return None


def eval_sell_save(sell, dir_name):
	sell = sell.reshape([-1, 3])
	df = pd.DataFrame(sell, columns=["sell_price", "time", "num"])
	df.to_csv("{}/eval/sell.csv".format(dir_name), index=False)
	return None


def eval_total_save(total, dir_name):
	total = total.reshape([-1, 1])
	df = pd.DataFrame(total, columns=["reward"])
	df.to_csv("{}/eval/reward.csv".format(dir_name, index=False))
	return None

def eval_graph(data, dir_name, time_len,start_time):

	buy = pd.read_csv("{}/eval/buy.csv".format(dir_name))
	sell = pd.read_csv("{}/eval/sell.csv".format(dir_name))
	fig=plt.figure(figsize=(20, 8))
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.plot(data.index[start_time:time_len+start_time], data["Price"][start_time:time_len+start_time], color='k')
	plt.scatter(buy["time"], buy["buy_price"], color='r')
	plt.scatter(sell["time"], sell["sell_price"], color='b')
	plt.title("buy_sell")
	plt.savefig("{}/eval/buy_sell_{}.png".format(dir_name,start_time))
	plt.close(fig)

	buy = pd.read_csv("{}/eval/buy.csv".format(dir_name))
	sell = pd.read_csv("{}/eval/sell.csv".format(dir_name))
	fig=plt.figure(figsize=(20, 8))
	plt.xlabel("Time")
	plt.ylabel("Num")
	plt.ylim(-100, 100)
	plt.bar(buy["time"], buy["num"], color='b')
	plt.bar(sell["time"], sell["num"], color='b')
	plt.title("num")
	plt.savefig("{}/eval/num_{}.png".format(dir_name,start_time))
	plt.close(fig)

	time_reward = pd.read_csv("{}/eval/reward.csv".format(dir_name))
	fig=plt.figure(figsize=(24, 8))
	plt.xlabel("Time")
	plt.ylabel("reward")
	plt.plot(data.index[start_time:time_len+start_time], time_reward["reward"], color='k')
	plt.title("reward")
	plt.savefig("{}/eval/reward_{}.png".format(dir_name,start_time))
	plt.close(fig)

	return None

if __name__ == '__main__':

	data = pd.read_csv("out_data.csv")
	state_size = data.shape[1]
	agent = Agent(state_size)
	agent_memory_append = agent.memory.append
	total_reward = np.array([[]])


	l = len(data) - 1  # 너무많으니까 줄여도됨
	l = 1440
	batch_size = 30
	episode_count = 200# 에피소드 반복횟수
	L = 100
	rwd_scaling = 10
	update_cycle = 5
	setting = "_rw10_2"
	#time_l,L,batch,episode
	#python train.py
	folder = make_folder(episode_count,setting)

	P = data["Price"]
	t1 = time.time()

	for e in range(episode_count + 1):
		print("Episode " + str(e) + "/" + str(episode_count))
		state = data.iloc[1, 1:].values.reshape(1, -1)
		state = np.insert(state, 3, [0], axis=1)
		total_profit = 0
		prev_num = 0
		action = 0
		buy_action = np.array([[]])
		sell_action = np.array([[]])
		cha = -1

		for t in range(1, l):

			#before hold
			if action == 0 or cha == 0:
				reward = (prev_num * (P[t] - P[t - 1]))/rwd_scaling
				total_profit += reward*rwd_scaling

			if action != 0 and cha !=0:
				reward = (prev_num * (P[t] - P[t - 1]) - cha*P[t]*0.0005)/rwd_scaling
				total_profit += reward*rwd_scaling

			Q_values = agent.act(state)
			action = np.argmax(Q_values)
			num = Q_values[0][action] * L


			# hold
			next_state = data.iloc[t + 1, 1:].values.reshape(1, -1)
			next_state = np.insert(next_state, 3, [prev_num/100], axis=1)

			if action == 2:
				num = -1 * num

			if action != 0:
				# sell
				if prev_num > num:
					cha = prev_num - num
					prev_num = num
					if cha !=0:
						sell_action = np.append(sell_action, np.array([[P[t], t, -1*cha]]))
						#print("sell", action, round(reward, 2), round(cha, 3), round(num, 3))

				# buy
				else:
					cha = num - prev_num
					prev_num = num
					if cha !=0:
						buy_action = np.append(buy_action, np.array([[P[t], t, cha]]))
						#print("buy", action, round(reward, 2), round(cha, 3), round(num, 3))

			done = True if t == l - 1 else False
			agent_memory_append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print("--------------------------------")
				print("Total Profit: ", total_profit)
				print("--------------------------------")
				total_reward = np.append(total_reward, np.array([[total_profit, e]]))

			if len(agent.memory) > batch_size:
				agent.exp_replay(batch_size)

		if e % 5 == 0:
			agent.model.save("{}/models/model_ep".format(folder) + str(e))

		if e % update_cycle == 0:
			agent.update_target_from_model()

		buy_save(buy_action, folder, e)
		sell_save(sell_action, folder, e)

	#loss = agent.loss

	total_save(total_reward, folder)

	t2 = time.time()
	sec = (t2 - t1)
	check_time(sec, folder)
	graph(data, folder, l, episode_count)

	#eval

	model_name = folder + "/models" + "/model_ep" + str(episode_count)
	model = load_model(model_name)
	agent = Agent(state_size, True, model_name)
	start_time = 1440
	for_l = l+start_time

	state = data.iloc[1 + start_time, 1:].values.reshape(1, -1)
	state = np.insert(state, 3, [0], axis=1)
	total_profit = 0
	prev_num = 0
	action = 0
	buy_action = np.array([[]])
	sell_action = np.array([[]])
	total_array = np.array([[]])
	cha = -1

	for t in range(start_time, for_l):

		# before hold
		if action == 0 or cha == 0:
			reward = (prev_num * (P[t] - P[t - 1])) / rwd_scaling
			total_profit += reward * rwd_scaling
			total_array = np.append(total_array, total_profit)

		if action != 0 and cha != 0:
			reward = (prev_num * (P[t] - P[t - 1]) - cha * P[t] * 0) / rwd_scaling
			total_profit += reward * rwd_scaling
			total_array = np.append(total_array, total_profit)

		Q_values = agent.act(state)
		action = np.argmax(Q_values)
		num = Q_values[0][action] * L

		# hold
		next_state = data.iloc[t + 1, 1:].values.reshape(1, -1)
		next_state = np.insert(next_state, 3, [prev_num / 100], axis=1)

		if action == 2:
			num = -1 * num

		if action != 0:
			# sell
			if prev_num > num:
				cha = prev_num - num
				prev_num = num
				if cha != 0:
					sell_action = np.append(sell_action, np.array([[P[t], t, -1 * cha]]))
					#print("sell", round(reward, 2), round(cha, 3), round(num, 3))

			# buy
			else:
				cha = num - prev_num
				prev_num = num
				if cha != 0:
					buy_action = np.append(buy_action, np.array([[P[t], t, cha]]))
				# print("buy", action, round(reward, 2), round(cha, 3), round(num, 3))

		done = True if t == for_l - 1 else False
		agent_memory_append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: ", total_profit)
			print("--------------------------------")


	eval_buy_save(buy_action, folder)
	eval_sell_save(sell_action, folder)
	eval_total_save(total_array,folder)
	eval_graph(data, folder, l, start_time)