import argparse
import os
import copy
import logging
from src.sample import Select, txtProcess, Noise
from src.c_generate_soft import preprocess as c_preprocess
from src.classify import preprocess, download_albert

logging.basicConfig(level=logging.INFO)

categories = ['animal', 'company', 'film']
splits = ['train', 'valid', 'test']


def work(args):
    # we will create 3 sub-directory under classifier_dir to store files for classifier
    for category in categories:
        sub_dir = os.path.join(args.classifier_dir, category)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # we will create 3 sub-directory under generator_dir to store files for generator
    for category in categories:
        sub_dir = os.path.join(args.generator_dir, category)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # creat TopicList.txt under classifier_dir/category
    Select.work(args)

    # we assert that domains' data are under this data_dir's sub-directory
    # e.g: data_dir/animal/train.TitleText.txt
    for category in categories:
        for split in splits:
            if split == 'train':
                ignore_file = os.path.join(args.data_dir, category, 'ignoredIndices.log')
            else:
                ignore_file = os.path.join(args.data_dir, category, '%s_ignoredIndices.log' % split)

            data_paths = [os.path.join(args.data_dir, category, "%s.TitleText.txt" % (split)),
                          os.path.join(args.data_dir, category, "%s.src" % (split)),
                          os.path.join(args.data_dir, category, "%s.tgt" % (split)),
                          ignore_file
                          ]
            for data_path in data_paths:
                if not os.path.exists(data_path):
                    logging.info("[Error] No such path: {}".format(data_path))
                    logging.info(
                        "Please download dataset and put it under this directory: {}".format(args.data_dir))
                    quit()
    # filter some data from data_dir/category/split.TitleText.txt, creat split.TitleText.txt under classifier_dir/category
    txtProcess.work(args)

    new_args = copy.deepcopy(args)
    for category in categories:
        new_args.data_dir = os.path.join(args.data_dir, category)
        new_args.save_dir = os.path.join(args.generator_dir, category)
        # output text.pkl
        c_preprocess.work_text(new_args)

        new_args.data_dir = os.path.join(args.generator_dir, category, 'text.pkl')
        new_args.topic_dir = os.path.join(args.classifier_dir, category, 'TopicList.txt')
        new_args.save_dir = os.path.join(args.classifier_dir, category)
        # add noise data as a new topic
        Noise.work(new_args)
        # preprocess for classifier
        preprocess.work(new_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal_topic_num', type=int, default=5, help='topic number of animal domain')
    parser.add_argument('--classifier_dir', type=str, required=True, help='the root directory of classifier files')
    parser.add_argument('--generator_dir', type=str, required=True, help='the root directory of generator files')
    parser.add_argument('--data_dir', type=str, required=True, help='the directory of dataset')
    parser.add_argument('--tokenizer_dir', type=str, required=True, help='the directory of tokenizer')
    # Data parameters
    parser.add_argument('--max-len', type=int, default=100, help='max length of classify data')
    parser.add_argument('--noise-scale', type=int, default=2, help="noise data's scale / other topics' average scale")

    args = parser.parse_args()

    middle_dirs = [args.classifier_dir, args.generator_dir, args.tokenizer_dir]
    for middle_dir in middle_dirs:
        if not os.path.exists(middle_dir):
            os.mkdir(middle_dir)

    download_albert.tokenizer(args.tokenizer_dir)
    work(args)
