# coding: utf-8

import torch
import argparse
import os
import json
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(level=logging.INFO)
from collections import Counter
import numpy as np
from src.c_generate_soft.data import SOS, EOS

import spacy
from spacy.lang.en import English
spacy.prefer_gpu()

nlp = English() 
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def sentencize(paragraph_list):
    sents = []
    for para in paragraph_list:
        doc = nlp(para)
        for sent in doc.sents:
            t = sent.text
            if (len(t.split()) >= 5):
                sents.append(t)
        
    return sents


def tokenize(line, add_special_tokens=False):
    tokens = line.strip().split()
    if (add_special_tokens):
        return ['<s>'] + [token.lower() for token in tokens] + ['</s>']
    else:
        return [token.lower() for token in tokens]


def work_text(args):
    splits = ['train', 'valid', 'test']

    data = [[], [], []]
    length = [[], [], []]

    logging.info('[1] Processing data')
    for spi, sp in enumerate(splits):
        if sp == 'train':
            ignore_filename = os.path.join(args.data_dir, 'ignoredIndices.log')
        else:
            ignore_filename = os.path.join(args.data_dir, '%s_ignoredIndices.log' %sp)

        ignore_list = set()
        with open(ignore_filename, 'r') as fin:
            for line in fin:
                ignore_list.add(int(line.strip()))

        document_filename = os.path.join(args.data_dir, '%s.src' %sp)
        abstract_filename = os.path.join(args.data_dir, '%s.tgt' %sp)
        title_sep = '<EOT>'
        doc_sep = '<EOP>'
        abs_sep = '<SNT>'

        titles = []
        doc_segs = []
        abs_segs = []
        doc_len = []
        abs_len = []

        with open(document_filename, 'r') as fin:
            lines = fin.readlines()
            for i, line in tqdm(enumerate(lines)):
                if (i in ignore_list):
                    continue

                content = line.split(title_sep)[1].strip()
                ll = sentencize(content.split(doc_sep))
                doc_lines = [tokenize(l) for l in ll]
                doc_segs.append(doc_lines)
                doc_len.append([len(l) for l in doc_lines])

        with open(abstract_filename, 'r') as fin:
            lines = fin.readlines()
            for i, line in tqdm(enumerate(lines)):
                if (i in ignore_list):
                    continue

                ll = line.split(abs_sep)
                abs_lines = [tokenize(l) for l in ll]
                abs_segs.append(abs_lines)
                abs_len.append([len(l) for l in abs_lines])

        assert len(doc_segs) == len(abs_segs)

        for doc_seg, abs_seg, docl, absl in zip(doc_segs, abs_segs, doc_len, abs_len):
            if (len(doc_seg) > 0) and (len(abs_seg) > 0):
                data[spi].append([doc_seg, abs_seg])
                length[spi].append([docl, absl])

    # save data with token text
    logging.info('[2] Save token text')
    text_datafile = os.path.join(args.save_dir, 'text.pkl')
    with open(text_datafile, 'wb') as fout:
        pickle.dump(data, fout)
        pickle.dump(length, fout)


def work_id(args):
    logging.info('[1] Load data, topic and labels')
    text_datafile = os.path.join(args.save_dir, 'text.pkl')
    with open(text_datafile, 'rb') as fin:
        text_data = pickle.load(fin)
        text_length = pickle.load(fin)

    label_datafile = os.path.join(args.save_dir, 'label.pkl')
    with open(label_datafile, 'rb') as fin:
        text_labels = pickle.load(fin)

    with open(args.topic_dir, 'rb') as fin:
        topics = json.load(fin)
        num_topics = len(topics)

    frecord = open('rec.txt', 'w')

    logging.info('[2] Load Glove')
    glove = pickle.load(open(args.glove, 'rb'))
    word_dim = len(glove['the'])
    logging.info('Word dim = %d' % word_dim)

    logging.info('[3] Gather Topics')
    topic_masks = [[], [], []]
    data = [[], [], []]
    length = [[], [], []]
    labels = [[], [], []]

    for spl in range(len(text_data)):
        for j in range(len(text_data[spl])): # j-th sample
            assert len(text_data[spl][j][0]) == len(text_labels[spl][j][0])
            new_data = [[] for i in range(num_topics)]
            new_length = [0 for i in range(num_topics)]

            for i, label in enumerate(text_labels[spl][j][0]):
                new_data[label].extend(text_data[spl][j][0][i])

                if (text_length[spl][j][0][i] == 0):
                    print(text_data[spl][j][0])
                    quit()

                new_length[label] += text_length[spl][j][0][i]

            new_length = np.array(new_length)

            mask = (new_length > 0)

            if (np.all(mask[:-1] == np.zeros((num_topics-1)))): # nonsense input
                continue
            else:
                data[spl].append([[], []])
                data[spl][-1][0] = new_data[:-1]
                data[spl][-1][1] = text_data[spl][j][1]

                length[spl].append([[], []])
                length[spl][-1][0] = new_length[:-1]
                length[spl][-1][1] = text_length[spl][j][1]

                labels[spl].append([[], []])
                labels[spl][-1] = text_labels[spl][j]

                # add 'STOP' topic (valid and test set only)
                if (spl == 0):
                    final_mask = np.append(mask[:-1], 0)
                else:
                    final_mask = np.append(mask[:-1], 1)
                topic_masks[spl].append(final_mask)

    for spl in range(len(topic_masks)):
        topic_masks[spl] = np.asarray(topic_masks[spl])

    logging.info('[4] Count word frequency')
    wtof = {}
    for j in range(len(data[0])): # j-th sample of train set
        for k in range(2): # 0: content, 1: summary
            for l in range(len(data[0][j][k])): # l-th topic
                for word in data[0][j][k][l]:
                    wtof[word] = wtof.get(word, 0) + 1
    __wtof = Counter(wtof).most_common(args.vocab_size)
    needed_words = { w[0]: w[1] for w in __wtof }
    
    logging.info('[5] Build vocab')
    itow = ['<pad>', '<unk>', '<s>', '</s>']
    wtoi = {'<pad>': 0, '<unk>': 1, '<s>': SOS, '</s>': EOS}
    for w in wtoi:
        glove[w] = np.zeros((word_dim, ))
    missing_word_neighbors = {}

    all_abs_lengths = []
    all_abs_num = []

    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(2): # 0: content, 1: summary

                if (k == 1): # add special tokens for summaries
                    for l in range(len(data[i][j][k])): # l-th sentence
                        data[i][j][k][l] = ['<s>'] + data[i][j][k][l] + ['</s>']

                    length[i][j][k] = [len(s) for s in data[i][j][k]]

                    all_abs_lengths.extend(length[i][j][k])
                    all_abs_num.append(len(length[i][j][k]))

                else: # truncate oversize topics in documents
                    for l in range(len(data[i][j][k])):
                        if (length[i][j][k][l] > args.max_len): 
                            data[i][j][k][l] = data[i][j][k][l][:args.max_len]
                            length[i][j][k][l] = args.max_len

                max_len = max([len(s) for s in data[i][j][k]]) # max length of sentences for padding
                if (max_len == 0): # No useful information?
                    print(data[i][j][0][-1])
                    print(data[i][j][1])
                    quit()

                for l in range(len(data[i][j][k])): # l-th sentence or topic
                    for m, word in enumerate(data[i][j][k][l]): # m-th word
                        if (word not in needed_words) and (word != '<s>') and (word != '</s>'):
                            word = '<unk>'
                        elif word not in wtoi:
                            itow.append(word)
                            wtoi[word] = len(wtoi)
                        data[i][j][k][l][m] = wtoi[word]
                        # Find neighbor vectors for those words not in glove
                        if word not in glove:
                            if word not in missing_word_neighbors:
                                missing_word_neighbors[word] = []
                            for neighbor in data[i][j][k][l][m-5:m+6]: # window size: 10
                                if neighbor in glove:
                                    missing_word_neighbors[word].append(glove[neighbor])
                    if (max_len > len(data[i][j][k][l])):
                        data[i][j][k][l] += [0]*int(max_len - len(data[i][j][k][l])) # padding l-th sentence

                data[i][j][k] = np.asarray(data[i][j][k], dtype='int32')
                length[i][j][k] = np.asarray(length[i][j][k], dtype='int32')

    logging.info('[6] Calculate vectors for missing words')
    for word in missing_word_neighbors:
        vectors = missing_word_neighbors[word]
        if len(vectors) > 0:
            glove[word] = sum(vectors) / len(vectors)
        else:
            glove[word] = np.zeros((word_dim, ))
    weight_matrix = np.vstack([glove[w] for w in itow])
    print('Shape of weight matrix:')
    print(weight_matrix.shape)

    all_abs_lengths = np.array(all_abs_lengths)
    frecord.write('%f %f\n' %(np.mean(all_abs_lengths), np.median(all_abs_lengths)))
    frecord.write('%f %f\n' %(np.mean(all_abs_num), np.max(all_abs_num)))

    frecord.close()

    logging.info('[7] Save token ids')
    id_datafile = os.path.join(args.save_dir, 'id.pkl')
    with open(id_datafile, 'wb') as fout:
        pickle.dump(data, fout)
        pickle.dump(length, fout)
        pickle.dump(labels, fout)
        pickle.dump(topic_masks, fout)
        pickle.dump(weight_matrix, fout)
        pickle.dump(wtoi, fout)
        pickle.dump(itow, fout)
        pickle.dump(wtof, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film'], default='animal')
    parser.add_argument('--data-dir', type=str, default='/data/wikicatsum')
    parser.add_argument('--save-dir', type=str, default='/data/new_dataset/text')
    parser.add_argument('--topic-dir', default='/data/classifier/animal/TopicList.txt')
    parser.add_argument('--glove', default='/data/glove.840B.300d.py36.pkl', help='pickle file of glove')
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--title-num', type=int, default=20)
    parser.add_argument('--topic-num', type=int, default=5)
    parser.add_argument('--target', type=str, choices=['text', 'id', 'all'], default='text')

    args = parser.parse_args()
    title_num = '{}_titles'.format(args.title_num)
    topic_num = '{}_topics'.format(args.topic_num)
    args.data_dir = os.path.join(args.data_dir, args.category)
    args.save_dir = os.path.join(args.save_dir, title_num)
    args.save_dir = os.path.join(args.save_dir, topic_num)
    args.save_dir = os.path.join(args.save_dir, args.category)
    args.topic_dir = '/data/classifier/%s/TopicList.txt' %args.category
    if not(os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    if (args.target == 'text') or (args.target == 'all'):
        work_text(args)

    if (args.target == 'id') or (args.target == 'all'):
        work_id(args)
