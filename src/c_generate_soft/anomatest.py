import argparse
import random
import shutil
import os
import torch
from torch import nn, optim
import numpy as np

from data import TopicDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--data_path', default='/data/text/animal/id.pkl', help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--label_path', default='/data/text/animal/label.pkl')
    parser.add_argument('--topic_path', default='/data/classifier/animal/TopicList.txt')

    args = parser.parse_args()

    train_loader = TopicDataset(args.data_path, args.label_path, args.topic_path, 'train')

    null_topics = 0
    all_topics = 0

    for doc, summ, doc_len, summ_len, doc_label, summ_label, avail_topic_mask in train_loader.gen_batch():
        for sl in summ_label:
            if (avail_topic_mask[sl] == 0):
                null_topics += 1
            all_topics += 1

    print('Null topic num: %d, all topic num: %d, percentage: %f' %(null_topics, all_topics, float(null_topics)/float(all_topics)))