from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DoraSet(Dataset):
    def __init__(self,cases=1,start=0,set='train',env='1'):
        self.set=set
        self.env=env
        folder='test' if set=='test' else 'train'
        folder=f'data/task{env}/{folder}/'
        if env=='1':
            ENVnum = 1000  #the number of environment in task1
        elif env=='2':
            ENVnum = 100
        elif env=='3':
            ENVnum = 1
        else:
            print(f'env is worng')

        self.Img_envs=[]
        for i in tqdm(range(ENVnum)):

            if env=='3':
                envPath=folder
            else:
                envPath = os.path.join(folder, f'{i + 1:05d}/')

            gain=np.load(envPath+'beamGains.npy')
            gain=np.transpose(gain,(0,2,1))
            gain=rearrange(gain,'b r t -> b (r t)')

            env_path = os.path.join(envPath, 'environment.png')
            Img_env = Image.open(env_path)
            transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
            Img_env=1-transform(Img_env)
            Img_env[Img_env!=0]=1 ## when there is a buliding, the value is 1
            Img_env=np.asarray(Img_env,dtype=int)
            Img_env=np.transpose(np.squeeze(Img_env),(1,0))

            loc=np.load(envPath+'locations.npy',allow_pickle=True)
            w=loc[0]
            h=loc[1]
            #print(f"w={w},h={h}")

            scale1=w/2/Img_env.shape[0] # The true scale of this map(environment)
            scale2=Img_env.shape[0]/w # Map into the pix for plotting
            #print(f'scale1={scale1}, scale2={scale2}')

            BS_loc=loc[2]
            #print(f'The true BS loc is {BS_loc/2} (unit: m)')
            ##---normization----##
            BS_loc[:,0]=BS_loc[:,0]/w
            BS_loc[:,1]=BS_loc[:,1]/h

            UE_loc=loc[3]
            UE_loc[:,0]=UE_loc[:,0]/w
            UE_loc[:,1]=UE_loc[:,1]/h

            if i==0:
                self.gains=gain
                self.locations=np.concatenate((BS_loc,UE_loc),axis=1)
            else:
                self.gains=np.concatenate((self.gains,gain),axis=0)
                location=np.concatenate((BS_loc, UE_loc), axis=1)
                self.locations=np.concatenate((self.locations,location),axis=0)
            self.Img_envs.append(Img_env)

        for i in range(self.gains.shape[0]):#---normalization----#
            self.gains[i,:]=10**(self.gains[i,:]/10)/np.max(10**(self.gains[i,:]/10))

        self.gains=self.gains[start:start+cases,:]
        self.locations=self.locations[start:start+cases,:]

        if self.env=='1' or self.env=='2':
            self.Img_envs = self.Img_envs[start//10:(start//10+1+cases//10)] #there are 10 samples in each environment
        else:
            pass

    def __getitem__(self, idx):
        if self.env == '1' or self.env == '2':
            return self.gains[idx, :],idx // 10, self.locations[idx, :]
        else:
            return self.gains[idx, :], 0, self.locations[idx, :]

    def __len__(self):
        return len(self.gains)

if __name__=='__main__':
    dataset=DoraSet(cases=1,start=1000-100,set='valid',env='2')
    gains,Img_env_idx,locations=dataset[0]
    Img_env=dataset.Img_envs
    print(f'gains.shape={gains.shape}')
    #print(f'gains={gains}')
    print(f'Img_env.len={len(Img_env)}')
    print(f'Img_env.len={Img_env[Img_env_idx].shape}')
    print(f'locations.shape={locations.shape}')
    print(f'locations={locations}')
    f=plt.figure()
    for i in range(1):
        gain, Img_env, locations=dataset[i]
        plt.plot(gain,'b-o')
    plt.show()

