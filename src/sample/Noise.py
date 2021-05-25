# coding: utf-8

import argparse
import pickle
import os
import json
from tqdm import tqdm
import numpy as np
import logging
import re
import random

logging.basicConfig(level=logging.INFO)


def make_batches(seg, batch_size=16):
    n = len(seg)
    batches = []
    for i in range(0, n, batch_size):
        batch = [' '.join(line) for line in seg[i:i + batch_size]]
        batches.append(batch)

    return batches


def mark_noise(sent, topic_num):
    last_topic_i = topic_num - 1
    clust_name = "clust" + str(last_topic_i)
    return sent + "\t\t" + "NOISE" + "\t\t" + clust_name + "\n"


def getPatternList():
    patternList = []
    
    patternList.append(re.compile(r'html'))
    patternList.append(re.compile(r'urltoken'))
    patternList.append(re.compile(r'cookies'))
    patternList.append(re.compile(r'href'))
    
    patternList.append(re.compile(r'\[ details \]'))
    patternList.append(re.compile(r'automatically generated'))
    patternList.append(re.compile(r'\[ maps \]'))
    
    patternList.append(re.compile(r'copyright'))
    patternList.append(re.compile(r'©'))
    
    patternList.append(re.compile(r'\W\s\d+\spp'))

    return patternList


def work(args):
    logging.info('[1] Load topics')
    with open(args.topic_dir, 'r') as fin:
        topics = json.load(fin)
    topic_num = len(topics)
    logging.info("len(topics): {}".format(topic_num))

    logging.info('[2] Load data')
    with open(args.data_dir, 'rb') as fin:
        data = pickle.load(fin)

    logging.info('[3] Compile regex')

    patternList = getPatternList()

    logging.info('[4] Find noise')
    label = [[], [], []]

    splits = ['train', 'valid', 'test']
    for sp in range(3):
        dd = data[sp]
        logging.info("{} seg num: {}".format(splits[sp], len(dd)))
        noise_sentences = []
        for doc_seg, abs_seg in tqdm(dd):
            label_seg = [[], []]

            for doc_line in doc_seg:
                doc_sent = ' '.join(doc_line)
                for pattern in patternList:
                    searchObj = pattern.search(doc_sent)
                    if searchObj:
                        noise_sentences.append(doc_sent)
                        break

            for abs_line in abs_seg:
                abs_sent = ' '.join(abs_line)
                for pattern in patternList:
                    searchObj = pattern.search(abs_sent)
                    if searchObj:
                        noise_sentences.append(abs_sent)
                        break
        logging.info("{} noise sentence num: {}".format(splits[sp], len(noise_sentences)))

        fileName = "{}.TitleText.txt".format(splits[sp])
        save_path = os.path.join(args.save_dir, fileName)
        with open(save_path, 'r') as fin:
            lines = fin.readlines()
            old_dataset_size = len(lines)
            fin.close()
        logging.info("{} old_dataset_size: {}".format(splits[sp], old_dataset_size))
        random.shuffle(noise_sentences)
        with open(save_path, 'a') as fout:
            print("open fout")
            for i, raw_sent in enumerate(noise_sentences):
                avrage_size = old_dataset_size / (topic_num - 1)
                if i > avrage_size * args.noise_scale:
                    break
                sentence = raw_sent.strip()
                if (sentence == ''):
                    continue
                noise_sentence = mark_noise(sentence, topic_num)
                fout.write(noise_sentence)
            fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film'], default='animal')
    parser.add_argument('--data-dir', type=str, default='/data/text')
    parser.add_argument('--save-dir', type=str, default='/data/classifier')
    parser.add_argument('--model-dir', type=str, default='/data/classification_models')
    parser.add_argument('--tokenizer-dir', type=str,
                        default='/data/pretrained_models/albert_tokenizer')
    parser.add_argument('--topic-dir', type=str, default='/data/classifier')

    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--max-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--noise-scale', type=int, default=2)

    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.category, 'text.pkl')
    args.save_dir = os.path.join(args.save_dir, args.category)
    args.topic_dir = os.path.join(args.topic_dir, args.category, 'TopicList.txt')
    if not (os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    work(args)
