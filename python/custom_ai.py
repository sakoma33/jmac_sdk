# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
from mjx import Action, Observation, State
from torch import Tensor, nn, optim, utils

# from tensorflow import keras
# from tensorflow.keras import layers


class MLP(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.net(x.float())


class MLP_large(pl.LightningModule):
    def __init__(self, obs_size=4488, n_actions=181, hidden_size=544):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        return self.net(x.float())


class MLP_small(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=544):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        return self.net(x.float())


# # 学習データに対する処理
# class TrainNet(pl.LightningModule):

#     @pl.data_loader
#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

#     def training_step(self, batch, batch_nb):
#         x, t = batch
#         y = self.forward(x)
#         loss = self.lossfun(y, t)
#         results = {'loss': loss}
#         return results

# # ディレクトリのパスを指定
# dir_path = "json"
# data_set = []
# # ディレクトリ直下のファイルを取得
# files = glob.glob("./json/*")
# # print(len(files))
# for i, file in enumerate(files[:1000]):
#     with open(file) as f:
#         lines = f.readlines()
#         for line in lines:
#             state = State(line)
#             for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
#                 obs = Observation._from_cpp_obj(cpp_obs)
#                 # feature = obs.to_features(feature_name="mjx-small-v0")
#                 feature = obs.to_features(feature_name="mjx-large-v0").ravel()

#                 action = Action._from_cpp_obj(cpp_act)
#                 action_idx = action.to_idx()
#                 data = np.append(feature, action_idx)
#                 data_set.append(data)

# print(len(data_set))

# data_set = np.array(data_set)
# np.save("dataset/large-1000", data_set)
# # data_set = np.load("dataset/large.npy")
# print(str(data_set.shape))
# # データをCSVファイルに保存する
# # np.savetxt('dataset/large.csv', data_set, delimiter=',')


dir_path = "json"
batch_size = 100  # バッチサイズの設定
data_set = []

files = glob.glob("./json/*")
num_files = len(files)

for i in range(0, num_files, batch_size):
    batch_files = files[i:i+batch_size]

    batch_data = []
    for file in batch_files:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                state = State(line)
                for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
                    obs = Observation._from_cpp_obj(cpp_obs)
                    feature = obs.to_features(feature_name="mjx-large-v0").ravel()

                    action = Action._from_cpp_obj(cpp_act)
                    action_idx = action.to_idx()
                    data = np.append(feature, action_idx)
                    batch_data.append(data)

    batch_data = np.array(batch_data)
    data_set.append(batch_data)

data_set = np.concatenate(data_set)
np.save("dataset/large-1000", data_set)

# def ai():

#     X_nn_train = X[:70]
#     y_nn_train = y[:70]
#     X_nn_valid = X[70:85]
#     y_nn_valid = y[70:85]
#     X_nn_test = X[85:100]
#     y_nn_test = y[85:100]
#     normalizer = tf.keras.layers.Normalization(axis=-1)
#     normalizer.adapt(np.array(X[:85]))

#     def build_and_compile_model(norm):
#         model = keras.Sequential([
#             norm,
#             layers.Dense(64, activation='relu'),
#             layers.Dense(64, activation='relu'),
#             layers.Dense(1)
#         ])

#         model.compile(loss='mean_squared_error',
#                       optimizer=tf.keras.optimizers.Adam(0.001),
#                       # metrics=['mean_squared_error']
#                       )
#         return model

#     nn_model = build_and_compile_model(normalizer)
#     nn_model.summary()

#     history = nn_model.fit(
#         X_nn_train,
#         y_nn_train,
#         validation_data=(X_nn_valid, y_nn_valid),
#         verbose=1, epochs=1000)

#     def plot_loss(history):
#         plt.plot(history.history['loss'], label='loss')
#         plt.plot(history.history['val_loss'], label='val_loss')
#         # plt.plot(np.log(np.array(history.history['loss'])), label='loss')
#         # plt.plot(np.log(np.array(history.history['val_loss'])), label='val_loss')
#         # plt.ylim([-0.1, 0.25])
#         plt.xlabel('Epoch')
#         plt.ylabel('')
#         plt.legend()
#         plt.grid(True)

#     plot_loss(history)
