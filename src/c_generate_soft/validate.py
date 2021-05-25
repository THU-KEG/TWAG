import torch
import argparse
import os
import json
import shutil
from tqdm import tqdm
from src.c_generate_soft.model import BiGRUEncoder, DocumentDecoder, TopicDecodeModel, TopicGenerator
from src.c_generate_soft.data import TopicDataset
from src.c_generate_soft.call_rouge import call_rouge
from src.c_generate_soft.utils import setup_logger, MetricLogger, strip_prefix_if_present, convert_to_string
from difflib import SequenceMatcher


def strip_redundant_sentences(paragraph):
    all_sents = paragraph.split('\n')
    chosen_sents = []
    for sent in all_sents:
        flag = False
        for prev_sent in chosen_sents:
            s = SequenceMatcher(None, sent, prev_sent)
            ll = s.find_longest_match(0, len(sent), 0, len(prev_sent)).size

            if (ll * 2 >= len(sent)):
                flag = True
                break

        if not (flag):
            chosen_sents.append(sent)

    return '\n'.join(chosen_sents)


def validate(data, model, device, log_dir, itow, fast=True):
    model.eval()
    torch.cuda.empty_cache()

    ref_dir = os.path.join(log_dir, 'ref')
    sum_dir = os.path.join(log_dir, 'sum')

    if os.path.isdir(ref_dir):
        shutil.rmtree(ref_dir)
    os.mkdir(ref_dir)
    if os.path.isdir(sum_dir):
        shutil.rmtree(sum_dir)
    os.mkdir(sum_dir)

    with torch.no_grad():
        for batch_iter, batch in tqdm(enumerate(data.gen_batch()), desc="Validating", total=data.example_num):
            if fast and batch_iter % 10 != 0:
                continue

            doc, summ, doc_len, summ_len, summ_label, avail_topic_mask = [a.to(device) for a in batch]

            # print("doc_device", doc.device)
            pred, pred_len = model(doc, doc_len, avail_topic_mask)
            golden = convert_to_string(itow, summ, summ_len)
            pred = convert_to_string(itow, pred, pred_len)

            pred = strip_redundant_sentences(pred)

            with open(os.path.join(ref_dir, "%d_reference.txt" % (batch_iter)), 'w') as f:
                f.write(golden)
            with open(os.path.join(sum_dir, "%d_decoded.txt" % (batch_iter)), 'w') as f:
                f.write(pred)

    # Rouge155_obj = Rouge155(stem=True, tmp=os.path.join(log_dir, 'tmp'))
    # score = Rouge155_obj.evaluate_folder(sum_dir, ref_dir)
    try:
        score = call_rouge(log_dir)
    except PermissionError:
        print("[PermissionError] Please use `python -m src.c_generate_soft.call_rouge` to get real score")
        score = {}
        score['rouge_1_f_score'] = 2021
    return score


def vis_scores(scores):
    recall_keys = {'rouge_1_recall', 'rouge_2_recall', 'rouge_l_recall', 'rouge_su4_recall'}
    f_keys = {'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score'}
    if type(list(scores.values())[0]) == dict:
        for n in scores:
            if n == 'all':
                scores[n] = {k: scores[n][k] for k in f_keys}
            else:
                scores[n] = {k: scores[n][k] for k in recall_keys}
    else:
        scores = {k: scores[k] for k in f_keys}
    return json.dumps(scores, indent=4)


def work(args):
    device = 'cuda'

    if (args.test):
        data = TopicDataset(args.data_path, args.label_path, args.topic_path, 'test')
    else:
        data = TopicDataset(args.data_path, args.label_path, args.topic_path, 'valid')

    args.num_vocab = len(data.weight)  # number of words
    loaded = torch.load(args.ckpt)
    model_kwargs = loaded['kwargs']
    for k in model_kwargs:
        if hasattr(args, k) and getattr(args, k) is not None:
            model_kwargs[k] = getattr(args, k)
    loaded['state_dict'] = strip_prefix_if_present(loaded['state_dict'], prefix='module.')
    for k, v in model_kwargs.items():
        print(k, v)

    model = TopicDecodeModel(**model_kwargs)
    model.load_state_dict(loaded['state_dict'])
    model = model.to(device)

    ckpt_base = os.path.splitext(os.path.basename(args.ckpt))[0]
    data_base = os.path.splitext(os.path.basename(args.data_path))[0]
    if args.test:
        log_dir = os.path.join(os.path.dirname(args.ckpt), 'test_%s_%s' % (data_base, ckpt_base))
    else:
        log_dir = os.path.join(os.path.dirname(args.ckpt), 'val_%s_%s' % (data_base, ckpt_base))

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    else:
        print('log dir %s exists, be careful that we will overwrite it' % log_dir)

    scores = validate(data, model, device, log_dir, data.itow, fast=args.fast)
    with open(os.path.join(log_dir, 'scores.txt'), 'w') as f:
        for k, v in model_kwargs.items():
            f.write('%s: %s\n' % (k, str(v)))
        f.write(json.dumps(scores, indent=4))
    vis = vis_scores(scores)
    print(vis)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--fast', action="store_true")
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    """
    parser.add_argument('--data_path', default='/data/text/animal/id.pkl')
    parser.add_argument('--label_path', default='/data/text/animal/label.pkl')
    parser.add_argument('--topic_path', default='/data/classifier/animal/TopicList.txt')
    """
    parser.add_argument('--classifier_dir', type=str, required=True, help='the root directory of classifier files')
    parser.add_argument('--generator_dir', type=str, required=True, help='the root directory of generator files')
    parser.add_argument('--topic_path', type=str, help='the path of TopicList.txt')

    # decode parameters
    parser.add_argument('--min_dec_len', type=int, default=10)
    parser.add_argument('--max_dec_len', type=int, default=400)
    parser.add_argument('--beam_size', type=int, default=4)
    # topic parameters
    parser.add_argument('--title_num', type=int, default=20)
    parser.add_argument('--topic_num', type=int, default=5)
    args = parser.parse_args()

    """
    args.data_path = '/data/text/{}_titles/{}_topics/{}_3e-5/id.pkl'.format(args.title_num, args.topic_num,
                                                                            args.category)
    args.label_path = '/data/text/{}_titles/{}_topics/{}/label.pkl'.format(args.title_num, args.topic_num,
                                                                           args.category)
    args.topic_path = '/data/classifier/%s/TopicList.txt' % args.category
    """
    # set path
    args.data_path = os.path.join(args.generator_dir, args.category, 'id.pkl')
    # [Tips] This label_path will not be used finally, just leave it alone
    args.label_path = '/data/text/%s/rouge_label.pkl' % args.category
    if not os.path.exists(args.topic_path):
        args.topic_path = os.path.join(args.classifier_dir, args.category, 'TopicList.txt')

    work(args)

if __name__ == "__main__":
    main()
