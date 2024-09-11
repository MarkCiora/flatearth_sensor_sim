import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2,3)
        self.fc2 = nn.Linear(9,1)
        
    def forward(self, x):
        x = x.view(-1, 3, 2)
        print(x.shape)
        x = relu(self.fc1(x))
        print(x.shape)
        x = x.view(-1, 9)
        print(x.shape)
        x = self.fc2(x)
        return x
    
net = Network()

v = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12], dtype=torch.float32)

# v = v.view(2,6)
# print(v.shape)
# v = net(v)
# print(v)


# v1 = torch.tensor([1,2,3,4,5,6], dtype=torch.float32)
# v2 = torch.tensor([1,2,3,4], dtype=torch.float32)+6

# v1 = v1.view(2,3)
# v2 = v2.view(2,2)

# print(v1)
# print(v2)

# v = torch.cat((v1,v2), dim=1)
# print(v)

x1 = np.array([1,2,3,4])
x2 = np.array([4,5,6])

x = np.hstack((x1,x2)).ravel()
print(x1, x2, x)