# coding: utf-8

import torch
import pickle
import random
import json

SOS = 2
EOS = 3

class TopicDataset:
    def __init__(self, data_path, label_path, topic_path, split, shuffle=False, fraction=1):
        __dic = {'train': 0, 'valid': 1, 'test': 2}
        self.data_split = __dic[split]

        with open(data_path, 'rb') as fin:
            self.data = pickle.load(fin)[self.data_split]
            self.length = pickle.load(fin)[self.data_split]
            self.labels = pickle.load(fin)[self.data_split]
            self.topic_masks = pickle.load(fin)[self.data_split]
            self.weight = pickle.load(fin)
            self.wtoi = pickle.load(fin)
            self.itow = pickle.load(fin)
            self.wtof = pickle.load(fin)

        with open(topic_path, 'rb') as fin:
            topics = json.load(fin)
            self.num_topics = len(topics)-1 # exclude "NOISE"

        self.shuffle = shuffle
        self.example_num = int(len(self.data)*fraction)
        self.data = self.data[:self.example_num] 
        self.length = self.length[:self.example_num] 
        self.labels = self.labels[:self.example_num]


    def gen_batch(self, required_index=None):
        assert (len(self.data) == len(self.length))
        assert (len(self.data) == len(self.labels))
        all_examples = list(zip(self.data, self.length, self.labels, self.topic_masks))
        if (self.shuffle):
            random.shuffle(all_examples)

        for data, length, labels, topic_mask in all_examples:
            _doc = data[0]
            _abs = data[1]
            _doc_len = length[0]
            _abs_len = length[1]
            _abs_label = labels[1]
            _avail_topic_mask = topic_mask

            yield torch.LongTensor(_doc), torch.LongTensor(_abs), torch.LongTensor(_doc_len), torch.LongTensor(_abs_len), \
                  torch.LongTensor(_abs_label), torch.tensor(_avail_topic_mask)

        raise StopIteration


if __name__ == "__main__":
    t = TopicDataset('/data/text/animal/id.pkl', '/data/text/animal/label.pkl', 'train')

    for doc, abs, doc_len, abs_len, doc_label, abs_label in t.gen_batch():
        print(doc_label)
        quit()