# coding: utf-8

from tqdm import tqdm
from transformers import AlbertTokenizer
import argparse
import os
import torch


def work(args):
    tokenizer = AlbertTokenizer.from_pretrained(args.tokenizer_dir)
    splits = ['train', 'valid', 'test']
    
    data = [[], [], []]
    max_topicnum = 0

    for i, sp in enumerate(splits):
        data_file = os.path.join(args.save_dir, '%s.TitleText.txt' %(sp))
        with open(data_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in tqdm(lines):
                try:
                    segs = line.strip().split('\t\t')
                    sentence = segs[0]
                    title = segs[1]
                    topic = int(segs[2][5:])
                    max_topicnum = max(max_topicnum, topic)
                except:
                    print(line)
                    quit()

                tokenize_res = tokenizer(sentence,
                                        add_special_tokens=True,
                                        truncation=True,
                                        max_length=args.max_len,
                                        pad_to_max_length = True,
                                        return_attention_mask = True,
                                        return_tensors='pt')

                data[i].append((tokenize_res['input_ids'], tokenize_res['attention_mask'], torch.tensor([topic])))

    torch.save(data, os.path.join(args.save_dir, 'albert_data_ml%d_tn%d.pt' %(args.max_len, max_topicnum+1)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Albert classification model.')
    # Data parameters
    parser.add_argument('--max-len', type=int, default=100)

    # S/L parameters
    parser.add_argument('--category', type=str, default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--save-dir', type=str, default='/data/classifier/animal')
    parser.add_argument('--tokenizer-dir', type=str, default='/data/pretrained_models/albert_tokenizer')

    args = parser.parse_args()
    args.save_dir = '/data/classifier/%s' %(args.category)

    work(args)