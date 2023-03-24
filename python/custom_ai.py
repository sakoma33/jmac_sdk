# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
import glob
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from mjx import Action, Observation, State
from pytorch_lightning import Trainer
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, TensorDataset

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


# dataset = np.load("dataset/large-100.npy")
# print(dataset.shape)
# x = dataset[:, :-1]
# y = dataset[:, -1]

# dataset = TensorDataset(torch.Tensor(x), torch.LongTensor(y))
# loader = DataLoader(dataset, batch_size=2)

# model = MLP()
# trainer = pl.Trainer(max_epochs=1)
# trainer.fit(model=model, train_dataloaders=train_loader)
# torch.save(model.state_dict(), './model_shanten_100.pth')


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


# dir_path = "json"
# batch_size = 10  # バッチサイズの設定
# data_set = []

# files = glob.glob("./json/*")
# files = files[:200]
# num_files = len(files)

# for i in range(0, num_files, batch_size):
#     batch_files = files[i:i+batch_size]

#     batch_data = []
#     for file in batch_files:
#         with open(file) as f:
#             lines = f.readlines()
#             for line in lines:
#                 state = State(line)
#                 for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
#                     obs = Observation._from_cpp_obj(cpp_obs)
#                     feature = obs.to_features(feature_name="mjx-large-v0").ravel()
#                     # feature = obs.to_features(feature_name="mjx-small-v0").ravel()

#                     action = Action._from_cpp_obj(cpp_act)
#                     action_idx = action.to_idx()
#                     data = np.append(feature, action_idx)
#                     batch_data.append(data)

#     batch_data = np.array(batch_data)
#     data_set.append(batch_data)

# data_set = np.concatenate(data_set)
# np.save("dataset/large-200", data_set)

def int_to_binary_vector(num):
    binary = bin(num)[2:].zfill(8)  # 8桁の2進数表現を取得
    vector = np.array(list(binary)).astype(int)  # 1次元配列に変換
    return vector


def binary_vector_to_int(vector):
    binary_str = ''.join(vector.astype(str))  # 1次元配列を文字列に変換
    num = int(binary_str, 2)  # 2進数表現の文字列を10進数の整数値に変換
    return num


# 学習データに対する処理
class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results


# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'val_loss': loss}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        results = {'val_loss': avg_loss}
        return results


# テストデータに対する処理
class TestNet(pl.LightningModule):

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'test_loss': loss}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'test_loss': avg_loss}
        return results


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):

    def __init__(self, input_size=4488, hidden_size=554, output_size=181, batch_size=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    # New: 平均ニ乗誤差
    def lossfun(self, y, t):
        return F.mse_loss(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


if __name__ == "__main__":

    dataset = np.load("dataset/large-100.npy")
    print(dataset.shape)
    x = dataset[:, :-1]
    y = dataset[:, -1]
    # y = np.array([int_to_binary_vector(element) for element in y])
    y = np.eye(181)[y]           # one hot表現に変換

    print(x.shape)
    print(y.shape)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x, y)
    print(dataset[0])
    n_train = int(len(dataset) * 0.6)
    n_val = int(len(dataset) * 0.2)
    n_test = len(dataset) - n_train - n_val
    print(n_train, n_val, n_test)
    # ランダムに分割を行うため、シードを固定して再現性を確保
    torch.manual_seed(0)

    # データセットの分割
    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    # 再現性の確保
    torch.manual_seed(0)

    # インスタンス化
    net = Net()
    trainer = Trainer(max_epochs=10)

    # 学習の実行
    trainer.fit(net)

    trainer.test()
    # print(trainer.callback_metrics)
    torch.save(net, 'models/first_model.pth')

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
