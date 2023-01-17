import torch
from torch import nn

class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8 * 4 + 4, 256),# use 32 beams and 4 locations
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 4),
            nn.Sigmoid()
        )

    def forward(self, beamGains, BS_UE_loc, environment,env_idx, device):
        ## the environment information hasn't been used yet.
        inputInfo = torch.cat([beamGains, BS_UE_loc], -1)
        out = self.mlp(inputInfo.to(device))
        return out

class model2(nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8 * 4 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 4),
            nn.Sigmoid()
        )

    def forward(self, beamGains, BS_UE_loc, environment,env_idx, device):
        inputInfo = torch.cat([beamGains, BS_UE_loc], -1)
        out = self.mlp(inputInfo.to(device))
        return out

class model3(nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8 * 4 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 4),
            nn.Sigmoid()
        )

    def forward(self, beamGains, BS_UE_loc, environment,env_idx, device):
        inputInfo = torch.cat([beamGains, BS_UE_loc], -1)
        out = self.mlp(inputInfo.to(device))
        return out


def main():
    b = 10
    doraNet = model1()

    beamGains1 = torch.zeros((b, 8 * 4))
    BS_UE_loc1 = torch.zeros(b, 4)
    environment1 = torch.zeros(2, 2)

    beamGains2 = torch.zeros((b, 8 * 4))
    BS_UE_loc2 = torch.zeros(b, 4)
    environment2 = torch.zeros(2, 2)

    beamGains3 = torch.zeros((b, 8 * 4))
    BS_UE_loc3 = torch.zeros(b, 4)
    environment3 = torch.zeros(2, 2)
    env_idx1=[1]
    env_idx2=[2]
    env_idx3=[3]

    device = torch.device("cpu")
    out1= doraNet(beamGains1, BS_UE_loc1, environment1,env_idx1, device)
    print(out1.shape)


if __name__ == '__main__':
    main()









