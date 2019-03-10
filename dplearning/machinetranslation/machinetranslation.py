from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

#获取并查看数据
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)#导入原始数据集

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# print("X.shape:", X.shape[0,:])
# print("Y.shape:", Y.shape)
# print("Xoh.shape:", Xoh.shape)
# print("Yoh.shape:", Yoh.shape)
# index = 0
# print("Source date:", dataset[index][0])
# print("Target date:", dataset[index][1])
# print()
# print("Source after preprocessing (indices):", X[index])
# print("Target after preprocessing (indices):", Y[index])
# print()
# print("Source after preprocessing (one-hot):", Xoh[index])
# print("Target after preprocessing (one-hot):", Yoh[index])

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

# Attention机制，计算隐藏层状态ct
def one_step_attention(a, s_prev):  # a = (m, Tx, 2*n_a)
    s_prev = repeator(s_prev)  # (m, n_s) to (m, Tx, n_s)
    concat = concatenator([s_prev, a])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context

n_a = 32  # Bi-LSTM隐藏层大小
n_s = 64  # Post-LSTM隐藏层大小
post_activation_LSTM_cell = LSTM(n_s, return_state = True)  # 隐藏层大小ns=64
output_layer = Dense(len(machine_vocab), activation=softmax)  # 输出层大小为输出的机器词汇的大小

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape=(Tx, human_vocab_size))  # (10000, 30, 37)
    s0 = Input(shape=(n_s,), name='s0')  #
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(n_a,return_sequences=True),input_shape=(m,Tx, n_a*2))(X)
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context,initial_state=[s,c])
        out = output_layer(s)
        outputs.append(out)
    model = Model(inputs=[X,s0,c0],outputs=outputs)
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr=0.005,beta_1=0.9,beta_2=0.999,decay=0.01)  # 梯度计算函数
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
# 初始化
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)





