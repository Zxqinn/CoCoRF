import numpy as np
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from nltk import stem
from nltk.corpus import stopwords

# PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]
PAD_ID, UNK_ID = [0, 1]
P = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')

def cos_np(data1,data2):
    """numpy implementation of cosine similarity for matrix"""
    dotted = np.dot(data1,np.transpose(data2))
    norm1 = np.linalg.norm(data1,axis=1)
    norm2 = np.linalg.norm(data2,axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

def normalize(data):
    """normalize matrix by rows"""
    normalized_data = data/np.linalg.norm(data,axis=1).reshape((data.shape[0], 1))
    return normalized_data

def dot_np(data1,data2):
    """cosine similarity for normalized vectors"""
    return np.dot(data1, np.transpose(data2))



#######################################################################

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d'% (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asMinutes(s), asMinutes(rs))

#######################################################################
import nltk
try: nltk.word_tokenize("hello world")
except LookupError: nltk.download('punkt')
    
def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''      
    def convert_sent(sent, vocab, maxlen):
        
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        keep_alpha = re.sub(ka, ' ', sent)
        split_hump = re.sub(P, r'\1 \2', keep_alpha)
        lower = split_hump.lower().split()
        remove_stop = [w for w in lower if w not in english_stop_words]
        stemmed = [stemmer.stem(j) for j in remove_stop]
        tokens = stemmed
    
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len):
            idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len

    if type(sentence) is list:
        inds, lens = [], []
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            inds.append(idxes)
            lens.append(idx_len)
        return np.vstack(inds), np.vstack(lens)
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]
    

########################################################################


def firstPos(real, predict):
    pos = len(predict)
    for idx, val in enumerate(predict):
        if val in real:
            pos = idx
            break
    return pos


def recall(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(real))


def precision(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(len(predict))


def f_measure(real, predict):
    pre = precision(real, predict)
    rec = recall(real, predict)
    try:
        f = 2 * pre * rec / (pre + rec)
    except ZeroDivisionError:
        f = -1
    return f


def ACC(real, predict):
    """accuracy = intersect / min(topk, len(real))"""
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1
    return sum / float(min(len(predict), len(real)))


def MAP(real, predict):
    sum = 0.0
    cur = 1
    l = len(real)
    for id, val in enumerate(predict):
        if val in real:
            sum = sum + cur / (id+1)
            cur+=1
            if cur == l:
                break
    return sum / l

def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1: sum = sum + 1.0 / float(index + 1)
    return sum / float(min(len(predict), len(real)))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg


def query2list(raw):
    keep_alpha = re.sub(ka, ' ', raw)
    split_hump = re.sub(P, r'\1 \2', keep_alpha)
    lower = split_hump.lower().split()
    remove_stop = [w for w in lower if w not in english_stop_words]
    stemmed = [stemmer.stem(j) for j in remove_stop]
    return stemmed

###########################################################

class PositionlEncoding(nn.Module):

    def __init__(self, d_hid, n_position=100):
        super(PositionlEncoding, self).__init__()

        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)