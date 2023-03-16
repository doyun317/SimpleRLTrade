import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import random
from collections import deque
import os

## 실행 전 텐서플로우 세팅
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def softmax(a) :
    c = np.max(a) # 최댓값
    exp_a = np.exp(a-c) # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # hold, buy, sell
        self.memory = deque(maxlen=5000)
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model(model_name) if is_eval else self._model()
        self.model_target = load_model(model_name) if is_eval else self._model()
        self.update_target_from_model()
        self.loss = []

    def _model(self):
        _inputs = tf.keras.Input(self.state_size)
        h1 = layers.Dense(300, activation="relu")(_inputs)
        h2 = layers.Dense(200, activation="relu")(h1)
        h3 = layers.Dense(100, activation="relu")(h2)
        _outputs = layers.Dense(self.action_size, activation="linear")(h3)
        model = Model(_inputs, _outputs)
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model


    def update_target_from_model(self):
        #Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())


    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            a1 = random.random()
            a2 = random.uniform(0, 1 - a1)
            a3 = 1 - a1 - a2
            return np.array([[a1, a2, a3]])

        options = self.model.predict(state)
        return softmax(options)

    def exp_replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        st_predict = self.model.predict(states)
        nst_predict = self.model.predict(next_states)
        nst_predict_target = self.model_target.predict(next_states)

        nst_predict_max_index = np.argmax(nst_predict, axis=1) # leanring agent의 Q값중 큰 action
        one_hot_max_index = tf.one_hot(nst_predict_max_index, self.action_size)

        target = rewards + self.gamma * np.amax(nst_predict_target*one_hot_max_index,axis=1) # 미래
        target[done] = rewards[done]

        target_f = st_predict
        target_f[range(batch_size), actions] = target

        # Q(s', a)
        #target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) #미래
        # end state target is reward itself (no lookahead)
        #target[done] = rewards[done]

        # Q(s, a)
        #target_f = self.model.predict(states) #현재로 예측하고? 사실상 array 만들어주는 역할, q(s,a)

        # make the agent to approximately map the current state to future discounted reward
        #target_f[range(batch_size), actions] = target #Q(s', a) 값을 업데이트, argmaxQ(s_t+1,a)

        hist = self.model.fit(states, target_f, epochs=1, verbose=0) #현재 스테이트 넣고 계산된 미래 Q값을 학습시키는 것
        #print(hist.history['loss'])
        #self.loss.append(hist.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
