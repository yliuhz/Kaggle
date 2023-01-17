from model import model1, model2, model3
from util import *
from dataset import DoraSet
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
sys.path.append("../..")

def train_model(task_flag):
    ##------------setups-----------------##
    seed_everything(42)  # fixed seed
    cudaIdx = "cuda:0"  # GPU card index
    saveLossInterval = 1  # intervals to save loss
    saveModelInterval = 1  # intervals to save model
    lr = 3e-4  # learning rate
    num_workers = 0  # workers for dataloader
    evaluation = False  # evaluation only if True
    loadRUNcase = f'/task1/model100.pth'

    if task_flag == 1:  # hyper parameters for task1
        epochs = 100
        num_samples = 10000
        valid_samples = 1000
        batchSize = 100
        idx_tmp = [0, 10, 17, 27, 32, 42, 49, 59]  # the measured beam index at each rx
        beam_used_idx = []
        for i in range(4):
            for j in range(8):
                beam_used_idx.append(idx_tmp[j] + i * 64)
    elif task_flag == 2:  # hyper parameters for task2
        epochs = 100
        num_samples = 1000
        valid_samples = 100
        batchSize = 100
        idx_tmp = [0, 18, 36, 54, 65, 83, 101, 119]
        beam_used_idx = []
        for i in range(4):
            for j in range(8):
                beam_used_idx.append(idx_tmp[j] + i * 128)
    elif task_flag == 3:  # hyper parameters for task3
        epochs = 100
        num_samples = 1000
        valid_samples = 100
        batchSize = 100
        idx_tmp = [0, 10, 17, 27, 32, 42, 49, 59]  # the same beam index as task1
        beam_used_idx = []
        for i in range(4):
            for j in range(8):
                beam_used_idx.append(idx_tmp[j] + i * 64)
    else:
        print(f'The task_flag is WRONG!')

    if task_flag == 1:
        runCase = f'task1'
    elif task_flag == 2:
        runCase = f'task2'
    else:
        runCase = f'task3'
    makeDIRs(runCase)
    if not os.path.exists(f'models/task_all/'):
        os.makedirs(f'models/task_all/')

    ##-----------------------------------##
    print('Preparing valid dataset ...')
    valid_dataset = DoraSet(cases=valid_samples, start=num_samples - valid_samples, set='valid', env=str(task_flag))
    valid_dataset_envs = valid_dataset.Img_envs
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    print('Valid dataset is ready!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if task_flag == 1:
        model_tmp = model1().to(device)
    elif task_flag == 2:
        model_tmp = model2().to(device)
    else:
        model_tmp = model3().to(device)
    criterion = torch.nn.MSELoss().to(device)
    Valid_score = []
    Train_score = []
    if not evaluation:
        print('Preparing train dataset ...')
        train_dataset = DoraSet(cases=num_samples - valid_samples, start=0, set='train', env=str(task_flag))
        train_dataset_envs = train_dataset.Img_envs
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
                                                   num_workers=num_workers)
        print('Train dataset is ready!')
        optimizer = torch.optim.Adam(model_tmp.parameters(), lr=lr)
        if task_flag==3: # transfer learning for task3
            model_tmp.load_state_dict(torch.load(f'models{loadRUNcase}'))  # transfer learning

    else:
        models=torch.load(f'models/task_all/model.pth')
        if task_flag==1:
            model_tmp.load_state_dict(models['model1'])
        elif task_flag==2:
            model_tmp.load_state_dict(models['model2'])
        else:
            model_tmp.load_state_dict(models['model3'])
        optimizer = []

    for epoch in range(1, epochs + 1):
        with torch.no_grad():
            score_V = run(False, valid_dataset_envs, valid_loader, model_tmp, criterion, [], epoch, task_flag,
                          beam_used_idx,device)
        Valid_score.append(score_V)
        if not evaluation:
            score_T = run(True, train_dataset_envs, train_loader, model_tmp, criterion, optimizer, epoch, task_flag,
                          beam_used_idx,
                          device)
            Train_score.append(score_T)
            checkPoint(runCase, epoch, epochs, model_tmp, Train_score, Valid_score, saveModelInterval, saveLossInterval)
        else:
            break


def run(isTrain, dataset_envs, data_loader, model, criterion, optimizer, epoch, task_flag, beam_used_idx, device):
    model.train() if isTrain else model.eval()
    setStr = 'Train' if isTrain else 'Valid'
    losses = Recoder()
    cnt = 0
    score1 = 0
    score3 = 0
    score5 = 0
    dataset_envs = np.array(dataset_envs)  # all environment in the task
    for i, (gains, Img_env_idx, locations) in enumerate(data_loader):
        batch_size = gains.shape[0]
        gains = gains.float().to(device)
        locations = locations.float().to(device)
        batchsize = gains.shape[0]
        p_gains = model(gains[:, beam_used_idx], locations, dataset_envs, Img_env_idx, device)
        loss = criterion(p_gains, gains)
        if isTrain:
            backward(optimizer, loss)

        ##---------------cal score-------------------##
        for j in range(batchsize):
            cnt = cnt + 1
            _, p_1 = torch.topk(p_gains[j, :], 1, dim=-1)
            _, p_3 = torch.topk(p_gains[j, :], 3, dim=-1)
            _, p_5 = torch.topk(p_gains[j, :], 5, dim=-1)
            _, gt_1 = torch.topk(gains[j, :], 1, dim=-1)

            if gt_1 in p_1:
                score1 = score1 + 1
            if gt_1 in p_3:
                score3 = score3 + 1
            if gt_1 in p_5:
                score5 = score5 + 1

        losses.update(loss.item(), batch_size)
    ##---------------cal score-------------------##
    score = (score1 / cnt + score3 / cnt + score5 / cnt) / 3
    print(f'{setStr} Task{task_flag} Epoch: {epoch}---loss {loss.item():.4f}---(score1, score3, score5)=({score1 / cnt:.4f}, { score3 / cnt:.4f}, {score5 / cnt:.4f})---final score: {score:.4f}')

    return score


def main():

    print(f'==========================Start training for task1=================================')
    train_model(task_flag=1)
    print(f'==========================Start training for task2=================================')
    train_model(task_flag=2)
    print(f'==========================Start training for task3=================================')
    train_model(task_flag=3)

    ##----------To save three models in ONE model-------------------##
    model_tmp1 = model1()
    model_tmp2 = model2()
    model_tmp3 = model3()
    model_tmp1.load_state_dict(torch.load(f'models/task1/model100.pth'))
    model_tmp2.load_state_dict(torch.load(f'models/task2/model100.pth'))
    model_tmp3.load_state_dict(torch.load(f'models/task3/model100.pth'))
    torch.save({
        'model1': model_tmp1.state_dict(),
        'model2': model_tmp2.state_dict(),
        'model3': model_tmp3.state_dict(),
    }, f'models/task_all/model.pth', _use_new_zipfile_serialization=False)

    ##------------show how to load three models--------------------##
    models = torch.load(f'models/task_all/model.pth')
    model_tmp1.load_state_dict(models['model1'])
    model_tmp2.load_state_dict(models['model2'])
    model_tmp3.load_state_dict(models['model3'])

if __name__ == '__main__':
    main()
