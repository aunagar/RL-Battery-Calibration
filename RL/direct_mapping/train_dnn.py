import os
import numpy as np
import logging as log

import torch
import torch.utils.data as data
from dnn import BatteryData, SimpleNN

import matplotlib.pyplot as plt
plt.style.use('seaborn')

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')




def train(data_path, random_seed = 2020, training = "random", test_frac = 0.3):

    device = get_device()

    best_val_loss = 1e10

    train_data = BatteryData(data_path, split = "train", split_type = training,
                    test_frac = test_frac, seed = random_seed)
    val_data = BatteryData(data_path, split = "val", split_type = training, 
                    test_frac = test_frac, seed = random_seed)

    train_loader = data.DataLoader(train_data, batch_size = 128, shuffle = True)
    val_loader = data.DataLoader(val_data, batch_size = 128)

    model = SimpleNN(17, 4, [256, 256, 256, 1], device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, betas = (0.5, 0.999))

    no_improve = 0
    loss_fn = torch.nn.MSELoss()
    for i in range(30):
        total_loss = []

        model.train()
        model.requires_grad(True)

        for i_batch, (x, y) in enumerate(train_loader):

            inp, out = x.to(device), y.to(device)

            pred_y = model(inp)

            loss = loss_fn(pred_y, out)

            # grad zero
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.detach().item())
            total_loss.append(loss.detach().cpu())
        print("epoch {} average loss is {}".format(i, np.mean(total_loss)))
    
        val_loss = []
        model.eval()
        model.requires_grad(False)
        for v_batch, (x,y) in enumerate(val_loader):
            inp, out = x.to(device), y.to(device)

            pred_y = model(inp)

            loss = loss_fn(pred_y, out)

            val_loss.append(loss.item())

        print("epoch {} val loss is {}".format(i, np.mean(val_loss)))

        if np.mean(val_loss) < best_val_loss:
            print("saving the model ...")
            torch.save(model.state_dict(), "./checkpoints/best_fixed_05.pt")
            best_val_loss = np.mean(val_loss)
            no_improve = 0
        else:
            print("cost did not improve!, prev best cost {}".format(best_val_loss))
            no_improve += 1

        if (no_improve > 5):
            print("Cost did not improve for more than {} epochs".format(no_improve))
            print("breaking!")
            break

def test(path, random_seed = 2020, training = "random", test_frac = 0.3):

    device = get_device()
    test_data = BatteryData(path, split = "test", split_type = training,
                        test_frac = test_frac, seed = random_seed)
    test_loader = data.DataLoader(test_data, batch_size = 1024, shuffle = False)

    model = SimpleNN(17, 4, [256, 256, 256, 1], device)
    model.load_state_dict(torch.load("./checkpoints/best_q_split.pt", map_location = device))

    model.eval()
    model.requires_grad(False)
    gt_traj = []
    pred_traj = []
    for t_batch, (x, y) in enumerate(test_loader):
        inp, out = x.to(device), y.to(device)

        pred_y = model(inp)

        gt_traj += list(y.cpu().numpy().reshape(-1))
        pred_traj += list(pred_y.cpu().numpy().reshape(-1))
    error = np.linalg.norm(np.array(pred_traj) - np.array(gt_traj))/len(gt_traj)
    print("avg test errir is = {}".format(error))    
    x = np.linspace(0,len(gt_traj)-1,len(gt_traj))
    fig = plt.figure()
    plt.plot(x, pred_traj, linestyle = '', color='blue', label='Tracking', marker = 'o', markersize = 1)
    plt.plot(x, gt_traj, color='orange',linestyle='',label='Ground truth', marker = '.', markersize = 1)
    plt.ylim(2000, 8000)
    plt.xlabel('time')
    plt.ylabel('Qmax')
    plt.legend(loc="upper right", markerscale=3., scatterpoints=1, fontsize=10)
    plt.savefig('action_tracking_q.jpg')



if __name__ == "__main__":
    DATA_PATH = "/cluster/scratch/aunagar/RL-data/data_5511_trajectories_load_8_16_q_4000_7000_R_0.117215_0.117215_dt_1_short.npz"

    train(DATA_PATH, training= "fixed", test_frac= 0.5)
    # test(DATA_PATH, training = "q")