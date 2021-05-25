import torch
from torch import nn, optim
import numpy as np
from transformers import AlbertTokenizer, AlbertModel
from tensorboardX import SummaryWriter
# from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import os
from tqdm import tqdm
import random
import shutil

from src.classify.data import DataLoader
from src.classify.model import AlbertClassifierModel

import logging

logging.basicConfig(level=logging.INFO)
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def recursive_to_device(device, *tensors):
    return [recursive_to_device(device, *t) if isinstance(t, list) or isinstance(t, tuple) \
                else t.to(device) for t in tensors]


def train(args):
    logging.info('[1] Preparing data')
    dataLoader = DataLoader('train', args.data_dir, args.max_len, args.topic_num)
    valid_dataLoader = DataLoader('valid', args.data_dir, args.max_len, args.topic_num)
    test_dataLoader = DataLoader('test', args.data_dir, args.max_len, args.topic_num)

    logging.info('[2] Building model')

    model = AlbertClassifierModel(num_topics=args.topic_num, dropout=args.dropout,
                                  albert_model_dir=args.albert_model_dir)
    logging.info(model)
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        logging.info('Transfer models to cuda......')

    model = model.to(device)
    if (args.restore_model) or (args.validate) or (args.test):
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'bert_classifier.pkl')))
        logging.info('Loaded model!')

    logging.info('[3] Building optimizer and loss criterion')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params=params, lr=args.lr)
    train_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))

    criterion = nn.CrossEntropyLoss()

    if (args.validate):
        validate(args, valid_dataLoader, model, device, criterion)
        return

    if (args.test):
        validate(args, test_dataLoader, model, device, criterion)
        return

    logging.info('[4] Training')
    for epoch in range(args.epoch):
        model.train()
        all_losses = []
        all_precision = []
        for batch_iter, train_batch in enumerate(dataLoader.gen_batch()):
            progress = epoch + batch_iter / dataLoader.data_size

            input_ids, attention_mask, labels = train_batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            predict_y = model(input_ids, attention_mask, use_cls=False)
            groundtruth_y = labels.to(device)

            loss = criterion(predict_y, groundtruth_y)

            precision = int(torch.argmax(predict_y, 1) == groundtruth_y) / float(len(labels))

            if (torch.isnan(loss).item()):
                logging.warning('Bad loss: %f' % (loss.item()))
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(parameters=params, clip_value=args.clip_value)
            nn.utils.clip_grad_norm_(parameters=params, max_norm=args.clip_norm)
            optimizer.step()

            all_losses.append(loss.item())
            all_precision.append(precision)

            if batch_iter % 100 == 0:
                logging.info('Epoch %.2f, avg_loss: %.2f, avg_precision: %.2f, precision: %.2f' % (
                    progress, np.mean(all_losses), np.mean(all_precision), precision))
                logging.info('Predict_y: %s' % (str(predict_y)))
                logging.info('Groundtruth_y: %s' % (str(groundtruth_y)))
                all_losses = []
                all_precision = []

        logging.info('Validating epoch %d, total examples = %d' % (epoch, valid_dataLoader.data_size))
        validate(args, valid_dataLoader, model, device, criterion)

        try:
            os.mkdir(os.path.join(args.save_dir, 'checkpoints/' + str(epoch)))
        except:
            pass
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'checkpoints/' + str(epoch) + '/bert_classifier.pkl'))

    [rootLogger.removeHandler(h) for h in rootLogger.handlers if isinstance(h, logging.FileHandler)]


def validate(args, dataLoader, model, device, criterion):
    model.eval()
    total_train_loss = 0
    total_examples = 0
    total_batches = 0
    total_correct_examples = 0
    total_groundtruth_y = []
    total_pridict_y = []

    for batch_iter, valid_batch in tqdm(enumerate(dataLoader.gen_batch())):
        input_ids, attention_mask, labels = valid_batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        predict_y = model(input_ids, attention_mask, use_cls=False)
        groundtruth_y = labels.to(device)

        loss = criterion(predict_y, groundtruth_y)

        total_train_loss += loss.item()
        total_examples += len(labels)
        total_batches += 1
        total_correct_examples += int(torch.argmax(predict_y, 1) == groundtruth_y) / float(len(labels))

        # for micro & macro
        total_groundtruth_y.append(groundtruth_y)
        total_pridict_y.append(torch.argmax(predict_y, 1))

    """
    groundtruth_y_np = torch.stack(total_groundtruth_y).view(-1).cpu().numpy()
    pridict_y_np = torch.stack(total_pridict_y).view(-1).cpu().numpy()
    # precision
    logging.info("micro_precision: %.3f , macro_precision: %.3f  " % (
        precision_score(groundtruth_y_np, pridict_y_np, average='micro'),
        precision_score(groundtruth_y_np, pridict_y_np, average='macro')))
    logging.info("pricision for all classes:")
    logging.info(precision_score(groundtruth_y_np, pridict_y_np, average=None))
    # recall
    logging.info("micro_recall: %.3f , macro_recall: %.3f  " % (
        recall_score(groundtruth_y_np, pridict_y_np, average='micro'),
        recall_score(groundtruth_y_np, pridict_y_np, average='macro')))
    logging.info("recall for all classes:")
    logging.info(recall_score(groundtruth_y_np, pridict_y_np, average=None))
    # f1
    logging.info("micro_f1: %.3f , macro_f1: %.3f  " % (
        f1_score(groundtruth_y_np, pridict_y_np, average='micro'),
        f1_score(groundtruth_y_np, pridict_y_np, average='macro')))
    logging.info("f1 for all classes:")
    logging.info(f1_score(groundtruth_y_np, pridict_y_np, average=None))
    """
    # accuracy
    logging.info('Valid Result: Avg_loss: %.3f, Accuracy: %.3f' % (total_train_loss / float(total_batches),
                                                                   float(total_correct_examples) / float(
                                                                       total_examples)))


def prepare(args):
    if not (args.restore_model) and not (args.validate) and not (args.test):
        if os.path.isdir(args.save_dir):
            shutil.rmtree(args.save_dir)
        os.mkdir(args.save_dir)

    try:
        os.mkdir(os.path.join(args.save_dir, "checkpoints"))
    except:
        pass

    try:
        os.mkdir(args.albert_model_dir)
    except:
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    # make logging.info display into both shell and file
    if (os.path.exists(os.path.join(args.save_dir, 'stdout.log'))):
        os.remove(os.path.join(args.save_dir, 'stdout.log'))

    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Albert classification model.')
    # Data loading parameters
    parser.add_argument('--max-len', type=int, default=100)
    parser.add_argument('--topic-num', type=int, default=5)

    # Model parameters
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=114514)

    parser.add_argument('--clip-value', type=float, default=0.1, help='clip to prevent the too large grad')
    parser.add_argument('--clip-norm', type=float, default=1, help='clip to prevent the too large grad')

    # S/L parameters
    parser.add_argument('--category', type=str, default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--classifier-dir', type=str, required=True, help='the root directory of classifier files')
    parser.add_argument('--albert-model-dir', type=str, required=True,
                        help='the directory to store albert model file which will be downloaded from huggingface')
    parser.add_argument('--data-dir', type=str, default='/data/classifier/animal')
    parser.add_argument('--save-dir', type=str, default='/data/classification_models/classifier_animal')
    parser.add_argument('--model-dir', type=str,
                        default='/data/classification_models/classifier_animal/checkpoints/1')
    parser.add_argument('--restore-model', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    args.data_dir = os.path.join(args.classifier_dir, args.category)
    args.save_dir = os.path.join(args.classifier_dir, 'classifier_%s' % (args.category))

    prepare(args)
    train(args)
