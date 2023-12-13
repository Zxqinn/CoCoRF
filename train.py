import os
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import argparse
from collections import defaultdict
import torch.utils.data
import time
from torch.utils.data import dataset
import os.path as path
random.seed(42)
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


import torch
import torch.utils.data as tdata
import models, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from utils import normalize, dot_np
from data_loader import *

try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML, SESSION_NAME
except:
    IS_ON_NSML = False

def bind_nsml(model, **kwargs):
    if type(model) == torch.nn.DataParallel: model = model.module

    def infer(raw_data, **kwargs):
        pass

    def load(path, *args):
        weights = torch.load(path)
        model.load_state_dict(weights)
        logger.info(f'Load checkpoints...!{path}')

    def save(path, *args):
        torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))
        logger.info(f'Save checkpoints...!{path}')

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)


def train(args,start_time):
    # create file handler which logs even debug messages
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    def save_model(model, epoch):
        os.makedirs(args.save_path,exist_ok=True)
        torch.save(model.state_dict(),
                   path.join(args.save_path,f'epo{epoch}.h5'))

    def load_model(model, epoch, to_device):
        model.load_state_dict(torch.load(
            path.join(args.save_path,f'epo{epoch}.h5'),
            map_location=to_device))

    config = getattr(configs, 'config_' + args.model)()
    if args.automl:
        config.update(vars(args))
    print(config)

    ###############################################################################
    # Load data
    ###############################################################################
    data_path = args.data_path
    train_set = eval(config['dataset_name'])(data_path, config['train_name'],
                                             config['name_len'],
                                             config['train_code'],
                                             config['code_len'],
                                             config['train_desc'],
                                             config['desc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=config['batch_size'],
                                              shuffle=True, drop_last=True,
                                              num_workers=1)

    ###############################################################################
    # Define the models
    ###############################################################################
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)  # initialize the model
    if args.reload_from > 0:
        load_model(model, args.reload_from, device)
    if IS_ON_NSML:
        bind_nsml(model)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=config['learning_rate'],
                                  eps=config['adam_epsilon'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(data_loader) * config[
            'nb_epoch'])  # do not foget to modify the number when dataset is changed

    itr_global = args.reload_from + 1
    n_iters = len(data_loader)
    loss_file_path = args.save_path+'loss_log.txt'
    for epoch in range(int(args.reload_from / n_iters) + 1,
                       config['nb_epoch'] + 1):
        itr_start_time = time.time()
        losses = []
        for batch in data_loader:

            model.train()
            batch_gpu = [tensor.to(device) for tensor in batch]
            loss = model(*batch_gpu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            itr_global = itr_global + 1

            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                logger.info(
                    'dataset: %s epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                    (args.dataset, epoch, config['nb_epoch'], itr_global % n_iters, n_iters,
                     elapsed, np.mean(losses)))
                if IS_ON_NSML:
                    summary = {"summary": True, "scope": locals(),
                               "step": itr_global}
                    summary.update({'loss': np.mean(losses)})
                    nsml.report(**summary)

                losses = []
                itr_start_time = time.time()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        with open('{}/performance.txt'.format(args.save_path),'a') as loss_file:
            loss_file.write(f'Epoch {epoch}: Average Loss = {avg_loss}\n')

        if epoch == args.stop_epoch:
            save_model(model, epoch)
            break

def parse_args():
    parser = argparse.ArgumentParser(
        "Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./processed_dataset/attention_model/' ,
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder',
                        help='model name')
    parser.add_argument('--save_path', type=str, default='./output/attention_model')
    parser.add_argument('--dataset', type=str, default='github',
                        help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1,
                        help='epoch to reload from')

    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--automl', action='store_true', default=False,
                        help='use automl')
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=100,
                        help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=5000,
                        help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000,
                        help='interval to evaluation to concrete results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    # Model Hyperparameters for automl tuning
    parser.add_argument('--emb_size', type=int, default=-1,
                        help='embedding dim')
    parser.add_argument('--n_hidden', type=int, default=-1,
                        help='number of hidden dimension of code/desc representation')
    parser.add_argument('--lstm_dims', type=int, default=-1)
    parser.add_argument('--margin', type=float, default=-1)
    parser.add_argument('--stop_epoch', type=float, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    start_time = time.time()
    train(args,start_time)
