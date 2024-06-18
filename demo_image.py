import torch
from torch import nn, optim 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio
import scipy.io
import math
import random
dtype = torch.cuda.FloatTensor

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=3):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)    
        
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
        super(input_mapping, self).__init__()
        self.B = B
    def forward(self, input):
        x_proj = (input) @ self.B.t()
        in_data = torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
        return in_data
    
class INR(nn.Module): 
    def __init__(self,method):
        super(INR, self).__init__()
        
        if method == 'SIREN':
            mid = 450
            self.H_net = nn.Sequential(nn.Linear(3,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,1))
        elif method == 'PE':
            mid = 320
            self.B_gauss = torch.Tensor(mid,3).type(dtype)
            torch.nn.init.kaiming_normal_(self.B_gauss, a=math.sqrt(2))
            self.H_net = nn.Sequential(input_mapping(self.B_gauss),
                                    nn.Linear(2*mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,1))
            
        elif method == 'GKAN':
            mid = 260
            self.H_net = nn.Sequential(nn.Linear(3,mid),
                                        GKANLayer(mid,mid),
                                        GKANLayer(mid,mid),
                                        GKANLayer(mid,1))
        
    def forward(self, input):
        return (self.H_net(input))

def main():
    for method in ['PE','SIREN','GKAN']:
        seed=1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.cuda.empty_cache()

        data = "data/plane_2"
        max_iter = 2001
        lr_real = 0.001

        file_name = data+'gt.mat'
        mat = scipy.io.loadmat(file_name)
        gt_np = mat["Ohsi"][:,:,0:10]
        gt = torch.from_numpy(gt_np).type(dtype)

        [n_1,n_2,n_3] = gt.shape

        x_in = torch.arange(1,n_1+1)
        y_in = torch.arange(1,n_2+1)
        z_in = torch.arange(1,n_3+1)
        x_in,y_in,z_in = torch.meshgrid(
            x_in, y_in, z_in)
        x_in = torch.flatten(x_in).unsqueeze(1)
        y_in = torch.flatten(y_in).unsqueeze(1)
        z_in = torch.flatten(z_in).unsqueeze(1)
        in_crood = torch.cat((x_in,y_in,z_in),dim=1).type(dtype)

        model = INR(method).type(dtype)

        params = []
        params += [x for x in model.parameters()]

        s = sum([np.prod(list(p.size())) for p in params]); 
        print('Number of params: %d' % s)

        optimizier = optim.Adam(params, lr=lr_real, weight_decay=0) 

        ps_best = 0

        for iter in range(max_iter):
            print('\r', iter, ps_best, end='\r\r')
            X_Out_real = model(in_crood).reshape(gt.shape)
            loss = torch.norm(X_Out_real-gt,2)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

            if iter % 50 == 0 and iter >= 1800:
                ps_here = peak_signal_noise_ratio(gt_np, 
                                                  np.clip(X_Out_real.cpu(
                                                      ).clone().detach().numpy(),0,1))
                if ps_here > ps_best:
                    ps_best = ps_here
                    
                plt.imshow(X_Out_real.cpu().clone().detach().numpy())
                plt.show()
                    
main()