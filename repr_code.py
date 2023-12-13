import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import threading
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
import torch
import utils
from data_loader import CodeSearchDataset, save_vecs
import models, configs
import jsonlines
import pickle
import os.path as path


def MRR(firstpos, top=20):
    reciprocal = []
    for p in firstpos:
        reciprocal.append(0 if p >= top else 1 / (p + 1))
    return np.mean(reciprocal)


##### Compute Representation #####
def repr_code(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_' + args.model)()
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)  # initialize the model
    if args.reload_from > 0:
        model.load_state_dict(torch.load(path.join('output/attention_model', f'epo{args.reload_from}.h5'), map_location=device))

    model = model.to(device)
    model.eval()

    use_set = eval(config['dataset_name'])(f'{args.data_path}/', config['use_names'], config['name_len'],
                                            config['use_codes'], config['code_len'])
    print('use_set:', len(use_set))
    data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=args.batch_size, shuffle=False,
                                              drop_last=False, num_workers=1)

    vecs, n_processed = [], 0
    for batch in tqdm(data_loader):
        batch_gpu = [tensor.to(device) for tensor in batch]
        with torch.no_grad():
            reprs = model.code_encoding(*batch_gpu).data.cpu().numpy()
        reprs = reprs.astype(np.float32)
        reprs = utils.normalize(reprs)
        vecs.append(reprs)
        n_processed = n_processed + batch[0].size(0)
    return np.vstack(vecs), model


def faster_eval(args, code_vecs, model, n_results, false_target=None):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_' + args.model)()
    descs, true_results = [], []
    # query_csv = pd.read_csv(path.join(args.eval_path, 'query.csv'))
    # query_csv = query_csv[query_csv['Language'].isin(['Python'])]
    # query_csv = query_csv[query_csv['Relevance'] > 0]

    for args.eval_dataset in ['CodeSearchNet', 'CosQA']:
        if args.eval_dataset == 'CodeSearchNet':
            with open(path.join(args.eval_path, 'final_eval.jsonl'), 'r') as f:
                for line in tqdm(jsonlines.Reader(f)):
                    url = line['id']
                    #row = query_csv[query_csv['GitHubUrl'] == url]
                    #descs.append(row['query'].values[0])
                    descs.append(line['query'])
                    true_results.append([url])
        elif args.eval_dataset == 'CosQA':
            with open(path.join(args.eval_path, 'cosqa_test.jsonl'), 'r') as f:
                for line in tqdm(jsonlines.Reader(f)):
                    url = line['idx']
                    descs.append(line['doc'])
                    true_results.append([url])

    code_reprs = []
    for i in range(0, len(code_vecs), args.chunk_size):
        code_reprs.append(code_vecs[i:i + args.chunk_size])
    codebase_id = []
    with open(path.join(args.data_path, 'codebase_id.txt'), 'r') as f:
        ids = f.read().split('\n')
        for i in range(0, len(ids), args.chunk_size):
            codebase_id.append(ids[i:i + args.chunk_size])
    desc_vocab = pickle.load(open(path.join(args.data_path, 'vocab.desc.pkl'), 'rb'))
    pos = []
    print(len(descs))
    for ii in range(len(descs)):
        real = [str(i).strip() for i in true_results[ii]]
        desc, desc_len = utils.sent2indexes(descs[ii], desc_vocab, 30)
        desc = torch.from_numpy(desc).unsqueeze(0).to(device)
        desc_len = torch.from_numpy(desc_len).clamp(max=config['desc_len']).to(device)
        with torch.no_grad():
            desc_repr = model.desc_encoding(desc, desc_len).data.cpu().numpy()
        desc_repr = utils.normalize(desc_repr).astype(np.float32).T
        threads = []
        predict = []
        if false_target:
            new_code_reprs = [[]]
            new_codebase_id = [[]]
            for i in range(len(codebase_id)):
                for j in range(len(codebase_id[i])):
                    if codebase_id[i][j].strip() in real:
                        new_code_reprs[0].append(code_reprs[i][j])
                        new_codebase_id[0].append(codebase_id[i][j])
            for t in false_target:
                new_code_reprs[0].append(code_reprs[int(t / args.chunk_size)][t % args.chunk_size])
                new_codebase_id[0].append(codebase_id[int(t / args.chunk_size)][t % args.chunk_size])
        else:
            new_code_reprs = code_reprs
            new_codebase_id = codebase_id
        for j, code_reprs_chunk in enumerate(new_code_reprs):
            t = threading.Thread(target=eval_thread,
                                 args=(predict, desc_repr, code_reprs_chunk, j, n_results, new_codebase_id))
            threads.append(t)
            t.start()
        for t in threads:  # wait until all sub-threads finish
            t.join()

        predict.sort(reverse=True, key=lambda x: x[0])
        predict = predict[:n_results]
        predict = [i[1] for i in predict]
        temp_pos = utils.firstPos(real, predict)
        pos.append(temp_pos)
    mrr = MRR(pos, top=n_results)
    hit = sum([p < n_results for p in pos])
    return mrr, hit


def eval_thread(sims, desc_repr, code_reprs, i, n_results, codebase_id):
    # 1. compute similarity
    chunk_sims = np.dot(code_reprs, desc_repr)  # [pool_size x 1]
    chunk_sims = np.squeeze(chunk_sims, axis=1)  # squeeze dim
    # chunk_sims = utils.dot_np(utils.normalize(desc_repr), code_reprs)
    # 2. choose top results
    negsims = np.negative(chunk_sims)
    maxinds = np.argpartition(negsims, kth=n_results - 1)
    maxinds = maxinds[:n_results]
    chunk_codes = [codebase_id[i][k] for k in maxinds]
    chunk_sims = chunk_sims[maxinds]
    sims.extend(zip(chunk_sims, chunk_codes))


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./processed_dataset/attention_model', help='location of the data corpus')
    parser.add_argument('--eval_path', type=str, default='./raw_dataset/python/eval', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('--datasource', type=str, default='CodeSearchNet', help='model name')
    parser.add_argument('--eval_dataset', type=str, default='CodeSearchNet', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github', help='dataset')
    parser.add_argument('--reload_from', type=int, default=100, help='epoch to reload from')
    parser.add_argument('--run_name', type=str, default='run_1642645207.5071788')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='how many instances for encoding and normalization at each step')
    parser.add_argument('--chunk_size', type=int, default=20000000,
                        help='split code vector into chunks and store them individually. '\
                             'Note: should be consistent with the same argument in the search.py')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(14)
    # false_target = random.sample(range(20604), 999)  # cosqa
    false_target = random.sample(range(23569), 999)  # csn
    vecs, model = repr_code(args)
    mrr, hit = faster_eval(args, vecs, model, 10, false_target=false_target)
    print(mrr, hit)
