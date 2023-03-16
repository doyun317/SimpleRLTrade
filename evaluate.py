from agent.agent import Agent
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def eavl_buy_save(buy, dir_name):
	buy = buy.reshape([-1, 3])
	df = pd.DataFrame(buy, columns=["buy_price", "time", "num"])
	df.to_csv("{}/eval/buy.csv".format(dir_name), index=False)
	return None


def eval_sell_save(sell, dir_name):
	sell = sell.reshape([-1, 3])
	df = pd.DataFrame(sell, columns=["sell_price", "time", "num"])
	df.to_csv("{}/eval/sell.csv".format(dir_name), index=False)
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

	return None

if __name__ == '__main__':
	l = 1000
	batch_size = 30
	episode_count = 40 # 에피소드 반복횟수
	L = 100
	rwd_scaling = 100
	update_cycle = 5
	setting = "_1"

	dir_name = str(datetime.date.today()) + '_rs' + str(rwd_scaling) + '_l' + str(l) + '_b' + str(
		batch_size) + '_e' + str(episode_count) + setting

	model_name = dir_name+"/models"+"/model_ep"+str(episode_count-1)
	model = load_model(model_name)

	data = pd.read_csv("out_data.csv")
	state_size = data.shape[1]

	agent = Agent(state_size, True, model_name)
	agent_memory_append = agent.memory.append

	P = data["Price"]
	start_time = 1440
	for_l=l+start_time

	state = data.iloc[1 + start_time, 1:].values.reshape(1, -1)
	state = np.insert(state, 3, [0], axis=1)
	total_profit = 0
	prev_num = 0
	action = 0
	buy_action = np.array([[]])
	sell_action = np.array([[]])
	cha = -1

	for t in range(start_time, for_l):

		# before hold
		if action == 0 or cha == 0:
			reward = (prev_num * (P[t] - P[t - 1])) / rwd_scaling
			total_profit += reward * rwd_scaling

		if action != 0 and cha != 0:
			reward = (prev_num * (P[t] - P[t - 1]) - cha * P[t] * 0) / rwd_scaling
			total_profit += reward * rwd_scaling

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

	buy_save(buy_action, dir_name)
	sell_save(sell_action, dir_name)
	graph(data, dir_name, l, start_time)