import os
import sys
from typing import List, Dict

from utils.train_test_split import *
import torch
import numpy as np
import torch.utils.data as data

class SimpleNN(torch.nn.Module):

    def __init__(
        self,
        input_dim : int,
        n_layers : int,
        hidden_units : List[int],
        device = torch.device
    ):

        super(SimpleNN, self).__init__()

        self.input_dim = input_dim
        layers = []

        dim1 = input_dim

        for i in range(n_layers):
            layers.append(torch.nn.Linear(dim1, hidden_units[i]))
            layers.append(torch.nn.LeakyReLU())
            dim1 = hidden_units[i]

        self.model = torch.nn.Sequential(*layers)
        self.to(device)
    
    def forward(self, x):
        assert x.shape[-1] == self.input_dim, "input dimenstion mis-match {}".format(x.shape)

        output = self.model(x)

        return output
    
    def requires_grad(self, val):
        for param in self.parameters():
            param.requires_grad = val


class BatteryData(data.Dataset):

    def __init__(
        self,
        path : str,
        split : str = "train", 
        split_type : str = "random",
        param : str = 0, # 0 = Qmax, 1 = Ro,
        test_frac : float = 0.3,
        seed : int = 2020):

        super().__init__()

        self.reference_state = torch.Tensor([292.1, 1., 1., 1., 6840, 760, 4.5600e+03, 506.6667, 20, 
                                        292.1, 1., 1., 1., 6840, 760, 4.5600e+03, 506.6667])
        self.split = split
        self.seed = seed
        self.split_type = split_type
        self.test_frac = test_frac
        self.param = param
        data = self.load_data(path, split)
        print(data.keys())
        self.x, self.y = self.get_xy(data)
    
    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        x = x/self.reference_state.to(x)
        y = torch.Tensor(self.y[index])

        return x, y
    
    def __len__(self):
        return len(self.y)


    
    def load_data(self, path, split):

        data, data_size = load_data(path)
        if self.split_type == "random":
            train, test = data_random_split(data, data_size, test_frac = self.test_frac, seed = self.seed)
            if split == "test":
                return test
            else:
                train_size = len(train['X'])
                train, val = data_random_split(train, train_size, test_frac = 0.2, seed = self.seed)
                if split == "train":
                    return train
                else:
                    return val
        elif self.split_type == "fixed":
            train, test = data_fixed_split(data, data_size, test_frac = self.test_frac, from_behind = True)
            if split == "test":
                return test
            else:
                train_size = len(train['X'])
                train, val = data_fixed_split(train, train_size, test_frac = 0.2, from_behind = True)
                if split == "train":
                    return train
                else:
                    return val
        elif self.split_type == "q":
            train, test = data_q_split(data, data_size, q_train = (4000,6000))
            if split == "test":
                return test

            else:
                train_size = len(train['X'])
                train, val = data_random_split(train, train_size, test_frac = 0.2, seed = self.seed)
                if split == "train":
                    return train
                else:
                    return val
        else:
            raise "No such split_type  = {}".format(self.split_type)

    def get_xy(self, data):
        state = data['X']
        theta = data['theta']
        load = data['U']
        X = [x[:-1,:] for x in state]
        X_ = [x[1:, :] for x in state]
        W = [w[:-1,:] for w in load]

        x = np.concatenate([np.concatenate((X[i], W[i], X_[i]), axis = -1) for i in range(len(X))], 0)[:,np.newaxis,:]
        y = np.concatenate([t.reshape(-1,2)[:-1,self.param] for t in theta],0).reshape(-1,1,1)
        return x,y


if __name__ == "__main__":
    device = torch.device('cpu')
    smpl = SimpleNN(17, 4, [256, 256, 256, 1], device)
    smpl.load_state_dict(torch.load("./checkpoints/best_fixed.pt", map_location = device))
    # x = torch.randn(5,1,5)
    # y = smpl(x)

    DATA_PATH = "/cluster/scratch/aunagar/RL-data/data_5511_trajectories_load_8_16_q_4000_7000_R_0.117215_0.117215_dt_1_short.npz"

    battery = BatteryData(path = DATA_PATH, split = "test", split_type = "q")
    batteryLoader = data.DataLoader(battery, batch_size = 5, shuffle = True)
    for i, (x,y) in enumerate(batteryLoader):
        if i > 10:
            break
        y_pred = smpl(x)
        print(y, y_pred)
