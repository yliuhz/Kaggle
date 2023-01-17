# !/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import torch
import json
from model_service.pytorch_model_service import PTServingBaseService
from model import model1, model2, model3


class BeamManagementService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        print('--------------------init--------------------')
        self.model_name = model_name
        self.model_path = model_path
        print(f"model_name:{model_name}")
        print(f"model_path:{model_path}")

        dir_path = os.path.dirname(os.path.realpath(model_path))
        print(f"dir_path={dir_path}")

        self.file_name = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        models = torch.load(model_path, map_location=device)

        model_1 = model1().to(device)
        model_1.load_state_dict(models['model1'])
        model_1.eval()
        model_2 = model2().to(device)
        model_2.load_state_dict(models['model2'])
        model_2.eval()
        model_3 = model3().to(device)
        model_3.load_state_dict(models['model3'])
        model_3.eval()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3

    def _preprocess(self, data):
        print('--------------------preprocess--------------------')
        preprocessed_data = {}
        for file_name, file_content in data['all_data'].items():
            print(f"file_name={file_name}, file_content={file_content}")
            self.file_name = file_name
            data_record = []
            lines = file_content.read().decode()
            lines = lines.split('\n')
            for line in lines:  # read all instance in the .txt
                if len(line) > 1:
                    data_record.append(json.loads(line))
            preprocessed_data[file_name] = data_record
        return preprocessed_data

    def _inference(self, data):
        print('--------------------inference----------------------')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        filename_tmp = self.file_name.split("_")
        task_flag = filename_tmp[0]
        print(f'---The task is {task_flag}---')
        print(f'fliename={self.file_name}')
        data_tmp = data[self.file_name]
        data_tmp = data_tmp[0]
        beamGains = torch.Tensor(data_tmp['beamGains'])
        BS_UE_loc = torch.Tensor(data_tmp['BS_UE_loc'])
        environment = torch.Tensor(data_tmp['environment'])
        env_idx = 0  # only 1 instance in a .txt

        if task_flag == "task1":
            result = self.model_1(beamGains, BS_UE_loc, environment, env_idx, device)
        elif task_flag == "task2":
            result = self.model_2(beamGains, BS_UE_loc, environment, env_idx, device)
        elif task_flag == "task3":
            result = self.model_3(beamGains, BS_UE_loc, environment, env_idx, device)
        else:
            print(f"The readed filename is WRONG !!!")

        ##------ Here to change Codes to get the following result1 result3 result5 ------##
        result = result.detach().cpu().numpy().tolist()
        result_tmp = np.array(result)
        idx = np.argsort(-result_tmp)
        result1 = idx[:1].tolist()
        result3 = idx[:3].tolist()
        result5 = idx[:5].tolist()
        results_fin = {'1': result1, '3': result3, '5': result5}
        ##-------------------------------END------------------------------##

        print(f'result_fin={results_fin}')
        return results_fin

    def _postprocess(self, data):
        print('---------postprocess--------------')

        return data
