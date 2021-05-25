# coding: utf-8

import torch
import argparse
import pickle
import os
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

from transformers import AlbertTokenizer, AlbertModel

from src.c_generate_soft.model import AlbertClassifierModel


def make_batches(seg, batch_size=16):
    n = len(seg)
    batches = []
    for i in range(0, n, batch_size):
        batch = [' '.join(line) for line in seg[i:i + batch_size]]
        batches.append(batch)

    return batches


def work(args):
    logging.info('[1] Load topics')
    with open(args.topic_dir, 'r') as fin:
        topics = json.load(fin)

    logging.info('[2] Load model')
    model = AlbertClassifierModel(num_topics=len(topics), albert_model_dir=args.albert_model_dir)
    tokenizer = AlbertTokenizer.from_pretrained(args.tokenizer_dir)
    model.load_state_dict(torch.load(args.model_dir), False)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    logging.info('[3] Load data')
    with open(args.data_dir, 'rb') as fin:
        data = pickle.load(fin)

    logging.info('[4] Mark labels')
    label = [[], [], []]

    for sp in range(3):
        dd = data[sp]
        index = 0

        for doc_seg, abs_seg in tqdm(dd):
            label_seg = [[], []]

            if args.debug:
                for doc_line in doc_seg:
                    doc_sent = ' '.join(doc_line)

                    tokenize_res = tokenizer(doc_sent,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length=args.max_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

                    predict_y = model.predict(tokenize_res['input_ids'].to(device),
                                              tokenize_res['attention_mask'].to(device))
                    predicted_topic = torch.argmax(predict_y, 1).item()
                    label_seg[0].append(predicted_topic)

                    print(doc_sent)
                    print(predict_y)
                    print(topics[predicted_topic][0])

                for abs_line in abs_seg:
                    abs_sent = ' '.join(abs_line)
                    tokenize_res = tokenizer(abs_sent,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length=args.max_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

                    predict_y = model.predict(tokenize_res['input_ids'].to(device),
                                              tokenize_res['attention_mask'].to(device), include_noise=False)
                    predicted_topic = torch.argmax(predict_y, 1).item()
                    label_seg[1].append(predicted_topic)

                    print(abs_sent)
                    print(predict_y)
                    print(topics[predicted_topic][0])

            else:
                doc_batches = make_batches(doc_seg, args.batch_size)
                for batch in doc_batches:
                    tokenize_res = tokenizer(batch,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length=args.max_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

                    predict_y = model.predict(tokenize_res['input_ids'].to(device),
                                              tokenize_res['attention_mask'].to(device))
                    predicted_topic = torch.argmax(predict_y, 1).cpu().numpy()
                    label_seg[0].extend(list(predicted_topic))

                abs_batches = make_batches(abs_seg, args.batch_size)
                for batch in abs_batches:
                    tokenize_res = tokenizer(batch,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length=args.max_len,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

                    predict_y = model.predict(tokenize_res['input_ids'].to(device),
                                              tokenize_res['attention_mask'].to(device), include_noise=False)
                    # print(predict_y)
                    predicted_topic = torch.argmax(predict_y, 1).cpu().numpy()
                    label_seg[1].extend(list(predicted_topic))

            # print(label_seg)
            label[sp].append(label_seg)
            index += 1

    with open(os.path.join(args.save_dir, 'label.pkl'), 'wb') as fout:
        pickle.dump(label, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film'], default='animal')
    parser.add_argument('--data-dir', type=str, default='/data/text')
    parser.add_argument('--save-dir', type=str, default='/data/text/20_titles_1e-7')
    parser.add_argument('--model-dir', type=str, default='/data/classification_models')
    parser.add_argument('--tokenizer-dir', type=str, default='/data/pretrained_models/albert_tokenizer')
    parser.add_argument('--topic-dir', type=str, default='/data/classifier')
    parser.add_argument('--albert-model-dir', type=str, required=True,
                        help='the directory to store albert model file which will be downloaded from huggingface')

    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--max-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.category, 'text.pkl')
    args.save_dir = os.path.join(args.save_dir, args.category)
    args.topic_dir = os.path.join(args.topic_dir, args.category, 'TopicList.txt')
    args.model_dir = os.path.join(args.model_dir, 'classifier_%s' % args.category, 'checkpoints', '3',
                                  'bert_classifier.pkl')
    if not (os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)

    if not (os.path.exists(args.albert_model_dir)):
        os.makedirs(args.albert_model_dir)

    work(args)
