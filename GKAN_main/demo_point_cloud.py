import numpy as np
import torch
from torch import nn, optim 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage.metrics import normalized_root_mse
import math
import random
import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=3):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias = True)    
        
    def forward(self, input):
        return self.linear(torch.sin(self.omega_0 * input))
    
class GKANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features*3, out_features)  
        self.omega_0 = torch.Tensor([[1],[2],[3]]).type(
            dtype).repeat(1, in_features).reshape(1, in_features*3)
    def forward(self, input):
        return self.linear(torch.sin(self.omega_0*input.repeat(1,3)))
    
class input_mapping(nn.Module):
    def __init__(self, B):
        super().__init__()
        self.B = B
    def forward(self, x):
        x_proj = (x) @ self.B.t()
        in_data = torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
        return in_data
        
class INR(nn.Module):
    def __init__(self, method):
        super(INR, self).__init__()
        
        if method == 'PE':
            mid_channel = 210
            self.B_gauss = torch.Tensor(mid_channel,4).type(dtype)
            std=1
            torch.nn.init.kaiming_normal_(self.B_gauss, a=math.sqrt(std))
            self.net = nn.Sequential(input_mapping(self.B_gauss),
                                      nn.Linear(2*mid_channel, mid_channel),
                                      nn.ReLU(),
                                      nn.Linear(mid_channel, mid_channel),
                                      nn.ReLU(),
                                      nn.Linear(mid_channel, mid_channel),
                                      nn.ReLU(),
                                      nn.Linear(mid_channel, mid_channel),
                                      nn.ReLU(),
                                      nn.Linear(mid_channel, 1))
        
        elif method == 'SIREN':
            mid_channel = 210
            self.net = nn.Sequential(nn.Linear(4, mid_channel),
                                      SineLayer(mid_channel, mid_channel),
                                      SineLayer(mid_channel, mid_channel),
                                      SineLayer(mid_channel, mid_channel),
                                      SineLayer(mid_channel, 1))
        
        elif method == 'GKAN':
            mid_channel = 120
            self.net = nn.Sequential(nn.Linear(4, mid_channel),
                                      GKANLayer(mid_channel, mid_channel),
                                      GKANLayer(mid_channel, mid_channel),
                                      GKANLayer(mid_channel, mid_channel),
                                      GKANLayer(mid_channel, 1))
        
    def forward(self, x):
        return self.net(x)

    
def main():
    for stat in ['data/mario011']:
        
        seed=1
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        np.random.seed(seed)
        random.seed(seed) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        p_test = 0.0      
        data_name = stat+'R'
        R_np = np.load(data_name+'.npy')
        
        mask = np.random.rand(R_np.shape[0])
        ind_train = np.array(np.where(mask > p_test))[0,:]
        
        R_data = R_np[:, 0:3]
        R_data4 = np.ones([R_data.shape[0], 1])
        R_data = np.concatenate((R_data, R_data4), axis = 1)
        Ry_data = R_np[:, 3]
        R_norm_train = R_data[ind_train,:]
        Ry_norm_train = Ry_data[ind_train]
        
        data_name = stat+'G'
        G_np = np.load(data_name+'.npy')
        G_data = G_np[:, 0:3]
        G_data4 = 2*np.ones([G_data.shape[0], 1])
        G_data = np.concatenate((G_data, G_data4),axis = 1)
        Gy_data = G_np[:, 3]
        G_norm_train = G_data[ind_train,:]
        Gy_norm_train = Gy_data[ind_train]
        
        data_name = stat+'B'
        B_np = np.load(data_name+'.npy')
        B_data = B_np[:, 0:3]
        B_data4 = 3*np.ones([B_data.shape[0], 1])
        B_data = np.concatenate((B_data, B_data4),axis = 1)
        By_data = B_np[:, 3]
        B_norm_train = B_data[ind_train,:]
        By_norm_train = By_data[ind_train]
        
        x_norm_train = np.concatenate((R_norm_train, G_norm_train, B_norm_train), axis = 0)
        y_train = np.concatenate((Ry_norm_train, Gy_norm_train, By_norm_train), axis = 0)
        y_train = np.expand_dims(y_train, axis=1)
        y_data = np.concatenate((Ry_data, Gy_data, By_data), axis = 0)
        
        y_data = np.expand_dims(y_data, axis=1)
        
        model = INR('GKAN').type(dtype)
        
        x_torch = torch.from_numpy(x_norm_train).type(dtype)
        y_torch = torch.from_numpy(y_train).type(dtype)
        
        params = []
        params += [x for x in model.parameters()]

        s = sum([np.prod(list(p.size())) for p in params]); 
        print('Number of params: %d' % s)
        
        optimizier = optim.Adam(params, lr=0.002, weight_decay=0) 
        
        mse_best = 10e6
        
        for i in range(15001):
            print('\r', i, end='')
            X_Out = model(x_torch)
            
            loss = torch.norm((X_Out) - y_torch, 2)
            
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            
            if i % 50 == 0:
                
                Pred_lrtfr = model(torch.from_numpy(
                    x_norm_train).type(dtype)).clone().detach().cpu()
                
                mse_test = normalized_root_mse(Pred_lrtfr.numpy(), y_train)
                
                if mse_test < mse_best:
                    mse_best = mse_test
                  
                print(f' MSE: {mse_test}',mse_best)
                
                y_R = Pred_lrtfr[0:R_np.shape[0], 0]
                y_G = Pred_lrtfr[R_np.shape[0]:2*R_np.shape[0], 0]
                y_B = Pred_lrtfr[2*R_np.shape[0]:3*R_np.shape[0], 0]
                
                color = np.stack([y_R,y_G,y_B], 
                                  1)/255
                
                fig = plt.figure(figsize = (6,6))
                ax = fig.add_subplot(1,1,1, projection='3d')
                ax.scatter3D(R_np[:,0],
                              R_np[:,1], 
                              R_np[:,2], facecolors = np.clip(color,0,1)) 
                ax.view_init(elev=-86, azim=-86)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                plt.axis('off')
                plt.show()

main()
                 