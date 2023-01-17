import json
from einops import rearrange
from PIL import Image
from torchvision import transforms
from util import *

task_flag = 1  # choose 1,2,or 3 to generate test file for task1,2,3

if task_flag == 1:
    num_env = 1
    instance_per_env = 10  # 1
    num_beams = 8 * 4
    idx_tmp = [0, 10, 17, 27, 32, 42, 49, 59] # Fixed
    beam_used_idx = []
    for i in range(4):
        for j in range(8):
            beam_used_idx.append(idx_tmp[j] + i * 64)
    beamGains = torch.zeros(num_env, instance_per_env, num_beams)
    BS_UE_loc = torch.zeros(num_env, instance_per_env, 4)

    path = "data/task1/train/"
    if not os.path.exists("data_test/label"):
        os.makedirs("data_test/label")
    if not os.path.exists("data_test/test_input"):
        os.makedirs("data_test/test_input")
    if not os.path.exists("data_test/test_output"):
        os.makedirs("data_test/test_output")
    for i in range(num_env):
        envPath = os.path.join(path, f'{i + 1:05d}/')
        gain = np.load(envPath + 'beamGains.npy')
        gain = np.transpose(gain, (0, 2, 1))
        gain = rearrange(gain, 'b r t -> b (r t)')
        for j in range(gain.shape[0]):
            gain[j, :] = 10 ** (gain[j, :] / 10) / np.max(10 ** (gain[j, :] / 10))
        for j in range(instance_per_env):
            idx_maxgain = gain[j].argmax()
            filename = 'data_test/label/task1_label_' + str(j) + '.txt'
            with open(filename, 'w') as name:
                data = {'idx': idx_maxgain.tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)

        env_path = os.path.join(envPath, 'environment.png')
        Img_env = Image.open(env_path)
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        Img_env_bak = Img_env
        Img_env = 1 - transform(Img_env)
        Img_env[Img_env != 0] = 1
        Img_env = np.asarray(Img_env, dtype=float)
        Img_env = np.transpose(np.squeeze(Img_env), (1, 0))

        loc = np.load(envPath + 'locations.npy', allow_pickle=True)
        w = loc[0]
        h = loc[1]
        BS_loc = loc[2]
        # print(f'The true BS loc is {BS_loc/2} (unit: m)')
        ##---normalization----##
        BS_loc[:, 0] = BS_loc[:, 0] / w
        BS_loc[:, 1] = BS_loc[:, 1] / h

        UE_loc = loc[3]
        UE_loc[:, 0] = UE_loc[:, 0] / w
        UE_loc[:, 1] = UE_loc[:, 1] / h

        locations = np.concatenate((BS_loc, UE_loc), axis=1)

        beamGains[i, :, :] = torch.Tensor(gain[:, beam_used_idx])
        BS_UE_loc[i, :, :] = torch.Tensor(locations)
        # print(f"beamGains[i,:,:].shape={beamGains[i,:,:].shape}")
        # print(f"BS_UE_loc[i,:,:].shape={BS_UE_loc[i,:,:].shape}")
        # print(f"env:{Img_env.shape}")
        for j in range(instance_per_env):
            filename = 'data_test/test_input/task1_data_' + str(j) + '.txt'
            with open(filename, 'w') as name:
                data = {'beamGains': beamGains[i, j, :].detach().tolist(),
                        'BS_UE_loc': BS_UE_loc[i, j, :].detach().tolist(),
                        'environment': torch.Tensor(Img_env).detach().tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)
elif task_flag == 2:
    num_env = 1
    instance_per_env = 10
    num_beams = 8 * 4
    idx_tmp = [0, 18, 36, 54, 65, 83, 101, 119] # Fixed
    beam_used_idx = []
    for i in range(4):
        for j in range(8):
            beam_used_idx.append(idx_tmp[j] + i * 128)
    beamGains = torch.zeros(num_env, instance_per_env, num_beams)
    BS_UE_loc = torch.zeros(num_env, instance_per_env, 4)

    path = "data/task2/train/"
    if not os.path.exists("data_test/label"):
        os.makedirs("data_test/label")
    if not os.path.exists("data_test/test_input"):
        os.makedirs("data_test/test_input")
    if not os.path.exists("data_test/test_output"):
        os.makedirs("data_test/test_output")
    for i in range(num_env):
        envPath = os.path.join(path, f'{i + 1:05d}/')
        gain = np.load(envPath + 'beamGains.npy')
        gain = np.transpose(gain, (0, 2, 1))
        gain = rearrange(gain, 'b r t -> b (r t)')
        for j in range(gain.shape[0]):
            gain[j, :] = 10 ** (gain[j, :] / 10) / np.max(10 ** (gain[j, :] / 10))
        for j in range(instance_per_env):
            idx_maxgain = gain[j].argmax()
            filename = 'data_test/label/task2_label_' + str(j) + '.txt'
            with open(filename, 'w') as name:
                data = {'idx': idx_maxgain.tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)

        env_path = os.path.join(envPath, 'environment.png')
        Img_env = Image.open(env_path)
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        Img_env_bak = Img_env
        Img_env = 1 - transform(Img_env)
        Img_env[Img_env != 0] = 1
        Img_env = np.asarray(Img_env, dtype=float)
        Img_env = np.transpose(np.squeeze(Img_env), (1, 0))

        loc = np.load(envPath + 'locations.npy', allow_pickle=True)
        w = loc[0]
        h = loc[1]
        BS_loc = loc[2]
        # print(f'The true BS loc is {BS_loc/2} (unit: m)')
        ##---normization----##
        BS_loc[:, 0] = BS_loc[:, 0] / w
        BS_loc[:, 1] = BS_loc[:, 1] / h

        UE_loc = loc[3]
        UE_loc[:, 0] = UE_loc[:, 0] / w
        UE_loc[:, 1] = UE_loc[:, 1] / h

        locations = np.concatenate((BS_loc, UE_loc), axis=1)

        beamGains[i, :, :] = torch.Tensor(gain[:, beam_used_idx])
        BS_UE_loc[i, :, :] = torch.Tensor(locations)
        # print(f"beamGains[i,:,:].shape={beamGains[i,:,:].shape}")
        # print(f"BS_UE_loc[i,:,:].shape={BS_UE_loc[i,:,:].shape}")
        # print(f"env:{Img_env.shape}")
        for j in range(instance_per_env):
            filename = 'data_test/test_input/task2_data_' + str(j) + '.txt'
            with open(filename, 'w') as name:
                data = {'beamGains': beamGains[i, j, :].detach().tolist(),
                        'BS_UE_loc': BS_UE_loc[i, j, :].detach().tolist(),
                        'environment': torch.Tensor(Img_env).detach().tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)

else:
    num_env = 1
    instance_per_env = 10
    num_beams = 8 * 4
    idx_tmp = [0, 10, 17, 27, 32, 42, 49, 59]
    beam_used_idx = []
    for i in range(4):
        for j in range(8):
            beam_used_idx.append(idx_tmp[j] + i * 64)
    beamGains = torch.zeros(num_env, instance_per_env, num_beams)
    BS_UE_loc = torch.zeros(num_env, instance_per_env, 4)

    path = "data/task3/train/"
    if not os.path.exists("data_test/label"):
        os.makedirs("data_test/label")
    if not os.path.exists("data_test/test_input"):
        os.makedirs("data_test/test_input")
    if not os.path.exists("data_test/test_output"):
        os.makedirs("data_test/test_output")
    for i in range(num_env):
        envPath = path
        gain = np.load(envPath + 'beamGains.npy')
        gain = np.transpose(gain, (0, 2, 1))
        gain = rearrange(gain, 'b r t -> b (r t)')
        for j in range(gain.shape[0]):
            gain[j, :] = 10 ** (gain[j, :] / 10) / np.max(10 ** (gain[j, :] / 10))

        for k in range(instance_per_env):
            gaink = gain[k, :]  # the k-th instance
            idx_maxgain = gaink.argmax()
            filename = 'data_test/label/task3_label_' + str(k) + '.txt'
            with open(filename, 'w') as name:
                data = {'idx': idx_maxgain.tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)

        env_path = os.path.join(envPath, 'environment.png')
        Img_env = Image.open(env_path)
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        Img_env_bak = Img_env
        Img_env = 1 - transform(Img_env)
        Img_env[Img_env != 0] = 1
        Img_env = np.asarray(Img_env, dtype=float)
        Img_env = np.transpose(np.squeeze(Img_env), (1, 0))

        loc = np.load(envPath + 'locations.npy', allow_pickle=True)
        w = loc[0]
        h = loc[1]
        BS_loc = loc[2]
        # print(f'The true BS loc is {BS_loc/2} (unit: m)')
        ##---normization----##
        BS_loc[:, 0] = BS_loc[:, 0] / w
        BS_loc[:, 1] = BS_loc[:, 1] / h

        UE_loc = loc[3]
        UE_loc[:, 0] = UE_loc[:, 0] / w
        UE_loc[:, 1] = UE_loc[:, 1] / h

        for kk in range(instance_per_env):
            BS_loc1 = BS_loc[kk, :]
            BS_loc1 = np.resize(BS_loc1, (1, 2))
            UE_loc1 = UE_loc[kk, :]
            UE_loc1 = np.resize(UE_loc1, (1, 2))
            # print(f'BS_loc.shape={UE_loc.shape}')
            locations = np.concatenate((BS_loc1, UE_loc1), axis=1)

            beamGains[i, kk, :] = torch.Tensor(gain[kk, beam_used_idx])
            BS_UE_loc[i, kk, :] = torch.Tensor(locations)

        for j in range(instance_per_env):
            filename = 'data_test/test_input/task3_data_' + str(j) + '.txt'
            with open(filename, 'w') as name:
                data = {'beamGains': beamGains[i, j, :].detach().tolist(),
                        'BS_UE_loc': BS_UE_loc[i, j, :].detach().tolist(),
                        'environment': torch.Tensor(Img_env).detach().tolist()}
                jsonData = json.dumps(data, ensure_ascii=False)
                name.write(jsonData)




