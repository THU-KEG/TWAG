import argparse
import random
import shutil
import os
import torch
from torch import nn, optim
import numpy as np
from rouge import Rouge

from utils import setup_logger, MetricLogger, strip_prefix_if_present
from data import TopicDataset
from model import BiGRUEncoder, DocumentDecoder, TopicDecodeModel, TopicGenerator
from pointer_generator import Beam, PointerGenerator
from validate import validate

import pdb
import traceback


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class GuruMeditation(torch.autograd.detect_anomaly):
    def __init__(self):
        super(GuruMeditation, self).__init__()

    def __enter__(self):
        super(GuruMeditation, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(GuruMeditation, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)

            print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            print("┃ Software Failure. Press left mouse button to continue ┃")
            print("┃        Guru Meditation 00000004, 0000AAC0             ┃")
            print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            print(str(value))
            pdb.set_trace()


def train(args, logger):
    logger.info('[1] Loading data')

    train_loader = TopicDataset(args.data_path, args.label_path, args.topic_path, 'train',
                                shuffle=not (args.no_shuffle), fraction=args.fraction)
    valid_loader = TopicDataset(args.data_path, args.label_path, args.topic_path, 'valid')
    test_loader = TopicDataset(args.data_path, args.label_path, args.topic_path, 'test')
    args.num_vocab = len(train_loader.weight) 
    args.num_topics = train_loader.num_topics
    logger.info('length of train/valid/test per gpu: %d/%d/%d' % (
        len(train_loader.data), len(valid_loader.data), len(test_loader.data)))

    logger.info('[2] Building model')
    device = torch.device('cuda')
    model = TopicDecodeModel(num_topics=args.num_topics,
                             num_vocab=args.num_vocab,
                             dim_word=args.dim_word,
                             dim_h=args.dim_h,
                             max_dec_sent=args.max_dec_sent,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             min_dec_len=args.min_dec_len,
                             max_dec_len=args.max_dec_len,
                             beam_size=args.beam_size,
                             is_coverage=args.is_coverage,
                             repeat_token_tres=args.repeat_token_tres,
                             same_sent_tres=args.same_sent_tres
                             ).to(device)

    model_kwargs = {k: getattr(args, k) for k in
                    {'dim_word', 'dim_h', 'num_vocab', 'num_layers', 'num_topics', 'dropout',
                     'max_dec_sent', 'min_dec_len', 'max_dec_len', 'beam_size', 'is_coverage',
                     'repeat_token_tres', 'same_sent_tres'}
                    }

    logger.info('[3] Initializing word embeddings')
    with torch.no_grad():
        weight = torch.tensor(train_loader.weight).float().to(device)
        print(weight)
        print('Shape of "weight":', weight.shape)
        print('Shape of "encoder.word_lookup.weight":', model.encoder.word_lookup.weight.shape)
        model.encoder.word_lookup.weight.set_(weight)
        model.sent_decoder.word_lookup.weight.set_(weight)
        model.word_lookup.weight.set_(weight)

    logger.info(model)

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank,
    #         # this should be removed if we update BatchNorm stats
    #         broadcast_buffers=False,
    #     )

    meters = MetricLogger(delimiter="  ")
    if args.reload_ckpt:
        assert args.is_coverage
        logger.info("Reload ckpt. Use coverage as Stage 2. Remember to use a small lr.")
        loaded = torch.load(args.reload_ckpt)
        loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
        model.load_state_dict(loaded['state_dict'], strict=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.schedule_step, gamma=0.1)
    score = validate(valid_loader, model, device, args.save_dir, train_loader.itow, fast=True)
    logger.info('[4] Start training......')
    for epoch_num in range(args.max_epoch):
        model.train()

        for batch_iter, train_batch in enumerate(train_loader.gen_batch()):
            progress = epoch_num + batch_iter / train_loader.example_num

            doc, summ, doc_len, summ_len, summ_label, avail_topic_mask = [a.to(device) for a in train_batch]
            sentence_loss, coverage_loss = model(doc, doc_len, avail_topic_mask, summ, summ_len, summ_label)

            if epoch_num < args.start_cov_epoch:
                losses = args.w_sentence * sentence_loss
            else:
                losses = args.w_sentence * sentence_loss + args.w_cov * coverage_loss

            optimizer.zero_grad()
            losses.backward()

            if args.clip_value > 0:
                nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
            if args.clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            meters.update(loss=losses, sentence_loss=sentence_loss, coverage_loss=coverage_loss)

            if (batch_iter + 1) % (train_loader.example_num // 100) == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {prog:.2f}",
                            "{meters}",
                        ]
                    ).format(
                        prog=progress,
                        meters=str(meters),
                    )
                )

        score = 0.111
        logger.info("val")
        logger.info(score)
        save = {
            'kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()

        torch.save(save,
                   os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score)))

        score = validate(valid_loader, model, device, args.save_dir, train_loader.itow, fast=True)

def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--data_path', default='/data/text/animal/id.pkl',
                        help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--label_path', default='/data/text/animal/label.pkl')
    parser.add_argument('--topic_path', default='/data/classifier/animal/TopicList.txt')
    parser.add_argument('--save_dir', default='/data/generate_models/animal',
                        help='path to save checkpoints and logs')
    parser.add_argument('--reload_ckpt', help='reload a checkpoint file')

    # training parameters
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--start_cov_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--w_label', type=float, default=1, help='weight of label loss')
    parser.add_argument('--w_sentence', type=float, default=1, help='weight of sentence loss')
    parser.add_argument('--w_cov', type=float, default=1, help='weight of coverage')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, nargs='+', default=[1])
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optim', default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--clip_norm', type=float, default=2.0)
    parser.add_argument('--use_rl', action='store_true')

    # model parameters
    parser.add_argument('--model_type', default='abs', choices=['abs', 'ext', 'ext_abs'])
    parser.add_argument('--dim_word', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--dim_h', type=int, default=512, help='dimension of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--repeat_token_tres', type=int, default=5, help='thres of the filter in Beam')
    parser.add_argument('--same_sent_tres', type=int, default=0.8,
                        help='thres of the filter in model.py to filt similar sentence')

    parser.add_argument('--is_coverage', action='store_true')

    # decode parameters
    parser.add_argument('--min_dec_len', type=int, default=10)
    parser.add_argument('--max_dec_len', type=int, default=500)
    parser.add_argument('--max_dec_sent', type=int, default=15)
    parser.add_argument('--beam_size', type=int, default=4)

    # data parameters
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--fraction', type=float, default=1, help='fraction of training set reduction')

    args = parser.parse_args()

    args.data_path = '/data/text/%s/id.pkl' % args.category
    args.label_path = '/data/text/%s/rouge_label.pkl' % args.category
    args.topic_path = '/data/classifier/%s/TopicList.txt' % args.category
    if (args.reload_ckpt) and (args.is_coverage):
        args.save_dir = '/data/generate_models_2/%s' % args.category
    else:
        args.save_dir = '/data/generate_models/%s' % args.category

    if args.reload_ckpt:
        # override model-related arguments when reloading
        model_arg_names = {'dim_word', 'dim_h', 'num_layers'}
        print(
            'reloading ckpt from %s, load its model arguments: [ %s ]' % (args.reload_ckpt, ', '.join(model_arg_names)))
        loaded = torch.load(args.reload_ckpt)
        model_kwargs = loaded['kwargs']
        for k in model_arg_names:
            setattr(args, k, model_kwargs[k])
    return args


def main():
    # dir preparation
    args = parse_args()
    # seed setting
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("WikiGen", args.save_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(args, logger)


if __name__ == '__main__':
    main()
