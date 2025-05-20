import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm import tqdm
import os
import h5py

"""
GDN에 적합하도록 WaferDataset을 변환한 데이터셋 클래스
dataset_preprocessing.ipynb에서 생성하여, h5 파일 이용
"""
class WaferDataset(Dataset):
    def __init__(self, file_list, model_type='reconstruction', max_len = 320):
        self.file_list = file_list
        self.model_type = model_type
        self.max_seq_len = max_len

        self.data, self.labels = [], []
        self.masks = []
        self.lengths = []
        self.lotids = []
        self.wafer_numbers = []
        self.step_nums = []

        for file_path in tqdm(file_list, desc="Loading and merging data"):
            with h5py.File(file_path, 'r') as f:
                raw_data = f['data'][:].astype(float)
                T_i = raw_data.shape[0]
                C = raw_data.shape[1]
                # D = raw_data.shape[2]
                # print(T_i, C)
                # padding
                if T_i < self.max_seq_len:
                    padded = np.zeros((self.max_seq_len, C))
                    padded[:T_i] = raw_data
                    mask = np.zeros((self.max_seq_len,), dtype=np.float32)
                    mask[:T_i] = 1.0
                else : 
                    padded = raw_data[:self.max_seq_len]
                    mask = np.ones((self.max_seq_len,), dtype=np.float32)
                    
                self.data.append(torch.tensor(padded, dtype=torch.float32))
                self.masks.append(torch.tensor(mask, dtype=torch.float32))
                
                # self.labels.append(f['labels'][:])
                self.lengths.append(min(T_i, self.max_seq_len))
                
                self.lotids.extend(f['lotids'][:].astype(str))
                self.wafer_numbers.extend(f['wafer_numbers'][:].astype(str))
                self.step_nums.extend(f['step_num'][:])

        # self.all_data = np.concatenate(all_data, axis=0)
        # self.all_next_steps = np.concatenate(all_next_steps, axis=0)
        # self.all_labels = np.concatenate(all_labels, axis=0)
        # self.n_sensor = self.data.shape[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            'given': self.data[idx],           # (max_seq_len, C)
            'mask': self.masks[idx],           # (max_seq_len,)
            'length': self.lengths[idx],       # int
            # 'label': self.labels[idx],         # int
            'lotid': self.lotids[idx],
            'wafer_number': self.wafer_numbers[idx],
            'step_num': self.step_nums[idx],
        }

        if self.model_type == 'reconstruction':
            item['answer'] = self.data[idx]
        # elif self.model_type == 'prediction':
        #     item['answer'] = self.data[idx][1:]  # 예: 다음 스텝 예측 등
        return item

def get_dataloader(data_info, loader_params: dict):
    """
    GDN 모델에 적합한 DataLoader 생성 함수 (train/val split 포함)

    Args:
        data_info (dict): 'train_dir' 키 포함
        loader_params (dict): batch_size, use_val 등 포함

    Returns:
        tuple: (train_loader, val_loader or None, test_loader)
    """
    # 전체 train 파일 리스트
    train_files = sorted([
        os.path.join(data_info['train_dir'], f)
        for f in os.listdir(data_info['train_dir'])
        if f.endswith(".h5")
    ])

    # validation 포함 여부
    if loader_params['use_val']:
        val_files = sorted([
        os.path.join(data_info['val_dir'], f)
        for f in os.listdir(data_info['val_dir'])
        if f.endswith(".h5")
    ])

        val_dataset = WaferDataset(val_files)
        val_loader = DataLoader(val_dataset,
                                batch_size=loader_params['batch_size'],
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
    else:
        val_loader = None

    # test 파일 리스트 (필요시 별도 dir로 교체 가능)
    test_files = sorted([
        os.path.join(data_info['test_dir'], f)
        for f in os.listdir(data_info['test_dir'])
        if f.endswith(".h5")
    ])

    # Dataset & DataLoader
    train_dataset = WaferDataset(train_files)
    test_dataset = WaferDataset(test_files)

    train_loader = DataLoader(train_dataset,
                              batch_size=loader_params['batch_size'],
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=loader_params['batch_size'],
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, val_loader, test_loader
