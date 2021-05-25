# coding: utf-8

from tqdm import tqdm
import argparse
import os
import torch
import random


class DataLoader:
    def __init__(self, data_split, path, max_len=100, topicnum=5):
        self.data_split = data_split
        self.data_file = os.path.join(path, 'albert_data_ml%d_tn%d.pt' %(max_len, topicnum))

        t = {'train': 0, 'valid': 1, 'test': 2}
        self.data = torch.load(self.data_file)[t[data_split]]
        random.shuffle(self.data)
        self.data_size = len(self.data)
        self.topicnum = topicnum
    
    def gen_batch(self):
        for input_ids, attention_mask, labels in self.data:
            yield input_ids, attention_mask, labels

        raise StopIteration