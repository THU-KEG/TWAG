import argparse
import os
import copy
import logging
from src.c_generate_soft import preprocess as c_preprocess
from src.c_generate_soft import make_labels

logging.basicConfig(level=logging.INFO)


def work(args):
    new_args = copy.deepcopy(args)
    category = args.category

    new_args.data_dir = os.path.join(args.generator_dir, category, 'text.pkl')
    new_args.topic_dir = os.path.join(args.classifier_dir, category, 'TopicList.txt')
    new_args.save_dir = os.path.join(args.generator_dir, category)
    new_args.model_dir = os.path.join(args.classifier_dir, 'classifier_%s' % args.category, 'checkpoints', '3',
                                      'bert_classifier.pkl')
    new_args.max_len = args.max_len_albert

    # output label.pkl
    make_labels.work(new_args)

    new_args.max_len = args.max_len_document
    new_args.glove = args.glove_path
    # output id.pkl
    c_preprocess.work_id(new_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier_dir', type=str, required=True, help='the root directory of classifier files')
    parser.add_argument('--generator_dir', type=str, required=True, help='the root directory of generator files')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='the directory of tokenizer')
    parser.add_argument('--albert-model-dir', type=str, required=True,
                        help='the directory to store albert model file which will be downloaded from huggingface')
    parser.add_argument('--glove_path', type=str, required=True, help='pickle file path of glove')
    parser.add_argument('--category', type=str, choices=['animal', 'company', 'film'], default='animal')
    parser.add_argument('--debug', action='store_true', help='if debug, make_labels.py will print extra massages')
    # Data parameters
    parser.add_argument('--max-len-document', type=int, default=400, help='max length of document for generate stage')
    parser.add_argument('--max-len-albert', type=int, default=100, help='max length of input for AlbertClassifier')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--vocab-size', type=int, default=50000)


    args = parser.parse_args()

    middle_dirs = [args.classifier_dir, args.generator_dir, args.tokenizer_dir]
    for middle_dir in middle_dirs:
        if not os.path.exists(middle_dir):
            logging.info("[Error] directory {} doesn't exists".format(middle_dir))
            logging.info("Please make sure you have finished classify stage")

    work(args)
