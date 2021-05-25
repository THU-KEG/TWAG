import argparse
import os
import json
from src.c_generate_soft.pyrouge.rouge import Rouge155

USE_LOG_DIR = True


def call_rouge(log_dir):
    ref_dir = os.path.join(log_dir, 'ref')
    sum_dir = os.path.join(log_dir, 'sum')

    Rouge155_obj = Rouge155(stem=True, tmp=os.path.join(log_dir, 'tmp'))
    score = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)

    with open(os.path.join(log_dir, 'scores.txt'), 'w') as f:
        f.write(json.dumps(score, indent=4))

    print(score)
    return score


def work():
    if USE_LOG_DIR:
        parser = argparse.ArgumentParser()
        parser.add_argument('--log_dir', type=str, required=True, help='the root directory of sum and ref')
        args = parser.parse_args()
        call_rouge(args.log_dir)
        return

    parser = argparse.ArgumentParser()

    parser.add_argument('--generator_dir', type=str, required=True, help='the root directory of generator files')
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--topic-num', type=int, default=5, help='num of topics')
    parser.add_argument('--title-num', type=int, default=20, help='num of titles')

    args = parser.parse_args()

    topic_num = '{}_topics'.format(args.topic_num)
    title_num = '{}_titles'.format(args.title_num)
    log_dir = os.path.join(args.generator_dir, args.category, 'trained_generate_models', topic_num, title_num)
    call_rouge(log_dir)


if __name__ == '__main__':
    work()
