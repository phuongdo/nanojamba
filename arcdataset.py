import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np


@dataclass
class ARCDatasetConfig:
    data_dir: str = "ARC-AGI/data/training"
    max_sequence_len: int = 4096

class ARCDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.data_files = []
        self.max_sequence_len = config.max_sequence_len
        for f in os.listdir(self.data_dir):
            if f.endswith('.json'):
                file_path = os.path.join(self.data_dir, f)
                self.data_files.append(file_path)
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file in self.data_files:
            with open(file, 'r') as f:
                task = json.load(f)
                for demonstration in task['train']:
                    demo_input = np.array(demonstration['input'])
                    demo_output = np.array(demonstration['output'])
                for test_case in task['test']:
                    test_case_input = np.array(test_case['input'])
                    test_case_output = np.array(test_case['output'])
                #TODO add data augmentation   
                # if self.data_aug:
                #     if np.random.rand() < 0.5:
                #         demo_input = np.flip(demo_input, axis=0)
                #         demo_output = np.flip(demo_output, axis=0)
                #         test_case_input = np.flip(test_case_input, axis=0)
                #         test_case_output = np.flip(test_case_output, axis=0)
                x = np.hstack([
                    demo_input.flatten(),
                    demo_output.flatten(),
                    test_case_input.flatten(),
                    test_case_output.flatten(),
                ])
                # shift by one for next token prediction
                y = x[1:] + [0]
                # padding
                x = self.pad_sequence(x, self.max_sequence_len)
                y = self.pad_sequence(y, self.max_sequence_len)
                data.append((x, y))
        return data

    def pad_sequence(self, sequence, max_length):
        padded_sequence = np.zeros(max_length, dtype=sequence.dtype)
        length = min(len(sequence), max_length)
        padded_sequence[:length] = sequence[:length]
        return padded_sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.LongTensor(x), torch.LongTensor(y)
