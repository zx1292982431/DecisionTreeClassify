import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,128)
        self.fc4 = nn.Linear(128,256)
        self.fc5 = nn.Linear(256,128)
        self.fc6 = nn.Linear(128,64)
        self.fc7 = nn.Linear(64,32)
        self.fc8 = nn.Linear(32,out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc2(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc3(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc4(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc5(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc6(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc7(x))
        x = nn.Dropout(0.2)(x)
        x = torch.relu(self.fc8(x))

        return x

class MLP_mini(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(MLP_mini, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x