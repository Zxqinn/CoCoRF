<<<<<<< HEAD
import nlqf
import os
import os.path as path
import json
import re
import torch
import javalang
import pickle
import numpy as np
import collections
from tables import *
from nltk import stem
from nltk.corpus import stopwords, wordnet
from python_structured import python_query_parse, python_code_parse

P = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('codeStopWord'))
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')


def get_first_sentence(docstring):
    docstring = re.split(r'[.\n\r]', docstring.strip('\n'))[0]
    return docstring


def contain_non_English(line):
    return bool(re.search(r'[^\x00-\xff]', line))


def remove_code_comment(line):
    line = line.replace('\'', '\"')
    line = re.sub(r'\t*"\"\"[\s\S]*?\"\"\"\n+', '', line)
    line = re.sub(r'\t*#+.*?\n+', '', line)
    line = re.sub('\n \n', '\n', line)
    return line


def load_CSN(CSN_path):
    raw_comments = []
    raw_code = []
    for file in os.listdir(CSN_path):
        with open(path.join(CSN_path, file), 'r') as f:
            line_dicts = [json.loads(i) for i in f.readlines()]
            filter_dicts = [d for d in line_dicts if not contain_non_English(str(d))]
            for line in filter_dicts:
                raw_comments.append(get_first_sentence(line['docstring']))
                raw_code.append({
                    'func_name': line['func_name'].split('.')[0],
                    'original_string': remove_code_comment(line['original_string']),
                    'url': line['url']
                })
    return raw_comments, raw_code


def extract_tokens(code_tokens):
    tokens = python_code_parse(code_tokens)
    return ' '.join(tokens)


def extract_docstring(docstring):
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    docstring = python_query_parse(docstring)
    return ' '.join(docstring)


def extract_method_name(function_name):
    function_name = function_name.split('.')[-1]
    sub_name_tokens = remove_underscores(function_name)
    sub_name_tokens = ' '.join(sub_name_tokens)
    return sub_name_tokens


def remove_underscores(function_name):
    cleaned_name = re.sub(r'^_+|_+$', '', function_name)
    cleaned_name = re.sub(r'\d', '', cleaned_name)
    return cleaned_name.lower().split('_')


def generate_vocab(data):
    tokens = ' '.join(data).split(' ')
    count = collections.Counter(tokens)
    tokens_unique = ['<PAD>', '<UNK>'] + [tok for tok, _ in count.most_common(9998)]
    vocab = {token: i for i, token in enumerate(tokens_unique)}
    return vocab


def token2id(data, vocab):
    id_list = []
    for token_str in data:
        tokens = token_str.split()
        ids = []
        for token in tokens:
            ids.append(vocab.get(token, 1))
        if len(ids) == 0:
            ids.append(0)
        id_list.append(ids)
    return id_list


class Features(IsDescription):
    length = Int64Col()
    pos = Int64Col()


def save_features(data, name, dir_path):
    phrases = []
    pos_pointer = 0

    fea_f = open_file(os.path.join(dir_path, name), mode='w',
                      title='Data for deep code search')
    table = fea_f.create_table('/', 'indices', Features, "lalala")
    rows = table.row
    for item in data:
        phrases.extend(item)
        rows['length'] = len(item)
        rows['pos'] = pos_pointer
        pos_pointer += len(item)
        rows.append()
    table.flush()

    arr = fea_f.create_array('/', 'phrases', np.array(phrases))
    arr.flush()
    fea_f.close()


def save_vocab(vocab, name='', dir_path=''):
    os.makedirs(dir_path, exist_ok=True)
    f = open(os.path.join(dir_path, name), 'wb')
    pickle.dump(vocab, f)
    f.close()


def extract_features(doc, code_str, func_name, dir_path):
    docstrings = []
    method_names = []
    codes = []
    for d, c ,f in zip(doc, code_str, func_name):
        docstring = extract_docstring(d)
        cod = extract_tokens(c)
        method_name = extract_method_name(f)
        method_names.append(method_name)
        codes.append(cod)
        docstrings.append(docstring)

    file_name = ['methname', 'code', 'desc']
    vocabs = []
    for i, data_list in enumerate(
            [method_names, codes, docstrings]):
        vocab = generate_vocab(data_list)
        save_vocab(vocab, 'vocab.{}.pkl'.format(file_name[i]), dir_path)
        vocabs.append(vocab)

    for j, data_list in enumerate(
            [method_names, codes, docstrings]):
        train_id_list = token2id(data_list, vocabs[j])

        save_features(train_id_list, '{}.{}.h5'.format('train', file_name[j]),
                      dir_path)


def filtering(raw_comments, raw_code, word_vocab,vae_model):
    comments, idx = nlqf.rule_filter(raw_comments)
    print('after rule filter:', len(comments))
    code = [raw_code[i] for i in idx]
    queries, idx = nlqf.model_filter(comments, word_vocab, vae_model)
    print('after model filter:', len(queries))
    code = [code[i] for i in idx]
    return queries, code


def extract_test_features(codes, query_dir, save_dir):
    eval_queries = []
    with open(path.join(query_dir, 'final_eval.jsonl'), 'r',encoding='utf-8') as f:
        for row in f.readlines():
            eval_queries.append(json.loads(row))
    methname, code, desc = [], [], []
    for i in eval_queries:
        methname.append(extract_method_name(i['method_name']))
        code.append(extract_tokens(i['code']))

    for i in codes:
        methname.append(extract_method_name(i['func_name']))
        code.append(extract_tokens(i['original_string']))
    data_dic = {
        'methname': methname,
        'code': code
    }
    for name,d in data_dic.items():
        with open(os.path.join(save_dir,f'vocab.{name}.pkl'),'rb') as f:
            vocab = pickle.load(f)
        id_list = token2id(d, vocab)
        save_features(id_list, '{}.{}.CSN.h5'.format('use', name), save_dir)
    with open(path.join(save_dir,'codebase_id.txt'), 'w') as f:
        f.write('\n'.join([str(i['id']) for i in eval_queries]+ [str(i['url']) for i in test_code]))


if __name__ == '__main__':
    CSN_train_path = './raw_dataset/python/train/'
    CSN_test_path = './raw_dataset/python/test/'
    CSN_eval_path = './raw_dataset/python/eval/'
    vocab_path = 'data/python/word_vocab.json'
    model_path = 'main_model/attention_model/main.model'
    save_dir = './processed_dataset/python/attention_model/'
    print("start")
    raw_comments, raw_code = load_CSN(CSN_train_path)
    print('train set size:', len(raw_comments))
    with open(vocab_path,'r') as f:
        word_vocab = json.load(f)
    model = torch.load(model_path)

    queries, code = filtering(raw_comments,raw_code,word_vocab,model)
    
    urls = [co['url'] for co in code]
    with open(save_dir + 'urls.txt', 'w') as f:
        for url in urls:
            f.write(url + '\n')
    print('filtered_size', len(queries))

    extract_features(queries, [co['original_string'] for co in code], [co['func_name'] for co in code], dir_path=save_dir)

    _, test_code = load_CSN(CSN_test_path)
    print(len(test_code))
    extract_test_features(test_code, CSN_eval_path, save_dir)
=======
<<<<<<< HEAD
import nlqf
import os
import os.path as path
import json
import re
import torch
import javalang
import pickle
import numpy as np
import collections
from tables import *
from nltk import stem
from nltk.corpus import stopwords, wordnet
from python_structured import python_query_parse, python_code_parse

P = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('codeStopWord'))
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')


def get_first_sentence(docstring):
    docstring = re.split(r'[.\n\r]', docstring.strip('\n'))[0]
    return docstring


def contain_non_English(line):
    return bool(re.search(r'[^\x00-\xff]', line))


def remove_code_comment(line):
    line = line.replace('\'', '\"')
    line = re.sub(r'\t*"\"\"[\s\S]*?\"\"\"\n+', '', line)
    line = re.sub(r'\t*#+.*?\n+', '', line)
    line = re.sub('\n \n', '\n', line)
    return line


def load_CSN(CSN_path):
    raw_comments = []
    raw_code = []
    for file in os.listdir(CSN_path):
        with open(path.join(CSN_path, file), 'r') as f:
            line_dicts = [json.loads(i) for i in f.readlines()]
            filter_dicts = [d for d in line_dicts if not contain_non_English(str(d))]
            for line in filter_dicts:
                raw_comments.append(get_first_sentence(line['docstring']))
                raw_code.append({
                    'func_name': line['func_name'].split('.')[0],
                    'original_string': remove_code_comment(line['original_string']),
                    'url': line['url']
                })
    return raw_comments, raw_code


def extract_tokens(code_tokens):
    tokens = python_code_parse(code_tokens)
    return ' '.join(tokens)


def extract_docstring(docstring):
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    docstring = python_query_parse(docstring)
    return ' '.join(docstring)


def extract_method_name(function_name):
    function_name = function_name.split('.')[-1]
    sub_name_tokens = remove_underscores(function_name)
    sub_name_tokens = ' '.join(sub_name_tokens)
    return sub_name_tokens


def remove_underscores(function_name):
    cleaned_name = re.sub(r'^_+|_+$', '', function_name)
    cleaned_name = re.sub(r'\d', '', cleaned_name)
    return cleaned_name.lower().split('_')


def generate_vocab(data):
    tokens = ' '.join(data).split(' ')
    count = collections.Counter(tokens)
    tokens_unique = ['<PAD>', '<UNK>'] + [tok for tok, _ in count.most_common(9998)]
    vocab = {token: i for i, token in enumerate(tokens_unique)}
    return vocab


def token2id(data, vocab):
    id_list = []
    for token_str in data:
        tokens = token_str.split()
        ids = []
        for token in tokens:
            ids.append(vocab.get(token, 1))
        if len(ids) == 0:
            ids.append(0)
        id_list.append(ids)
    return id_list


class Features(IsDescription):
    length = Int64Col()
    pos = Int64Col()


def save_features(data, name, dir_path):
    phrases = []
    pos_pointer = 0

    fea_f = open_file(os.path.join(dir_path, name), mode='w',
                      title='Data for deep code search')
    table = fea_f.create_table('/', 'indices', Features, "lalala")
    rows = table.row
    for item in data:
        phrases.extend(item)
        rows['length'] = len(item)
        rows['pos'] = pos_pointer
        pos_pointer += len(item)
        rows.append()
    table.flush()

    arr = fea_f.create_array('/', 'phrases', np.array(phrases))
    arr.flush()
    fea_f.close()


def save_vocab(vocab, name='', dir_path=''):
    os.makedirs(dir_path, exist_ok=True)
    f = open(os.path.join(dir_path, name), 'wb')
    pickle.dump(vocab, f)
    f.close()


def extract_features(doc, code_str, func_name, dir_path):
    docstrings = []
    method_names = []
    codes = []
    for d, c ,f in zip(doc, code_str, func_name):
        docstring = extract_docstring(d)
        cod = extract_tokens(c)
        method_name = extract_method_name(f)
        method_names.append(method_name)
        codes.append(cod)
        docstrings.append(docstring)

    file_name = ['methname', 'code', 'desc']
    vocabs = []
    for i, data_list in enumerate(
            [method_names, codes, docstrings]):
        vocab = generate_vocab(data_list)
        save_vocab(vocab, 'vocab.{}.pkl'.format(file_name[i]), dir_path)
        vocabs.append(vocab)

    for j, data_list in enumerate(
            [method_names, codes, docstrings]):
        train_id_list = token2id(data_list, vocabs[j])

        save_features(train_id_list, '{}.{}.h5'.format('train', file_name[j]),
                      dir_path)


def filtering(raw_comments, raw_code, word_vocab,vae_model):
    comments, idx = nlqf.rule_filter(raw_comments)
    print('after rule filter:', len(comments))
    code = [raw_code[i] for i in idx]
    queries, idx = nlqf.model_filter(comments, word_vocab, vae_model)
    print('after model filter:', len(queries))
    code = [code[i] for i in idx]
    return queries, code


def extract_test_features(codes, query_dir, save_dir):
    eval_queries = []
    with open(path.join(query_dir, 'final_eval.jsonl'), 'r',encoding='utf-8') as f:
        for row in f.readlines():
            eval_queries.append(json.loads(row))
    methname, code, desc = [], [], []
    for i in eval_queries:
        methname.append(extract_method_name(i['method_name']))
        code.append(extract_tokens(i['code']))

    for i in codes:
        methname.append(extract_method_name(i['func_name']))
        code.append(extract_tokens(i['original_string']))
    data_dic = {
        'methname': methname,
        'code': code
    }
    for name,d in data_dic.items():
        with open(os.path.join(save_dir,f'vocab.{name}.pkl'),'rb') as f:
            vocab = pickle.load(f)
        id_list = token2id(d, vocab)
        save_features(id_list, '{}.{}.CSN.h5'.format('use', name), save_dir)
    with open(path.join(save_dir,'codebase_id.txt'), 'w') as f:
        f.write('\n'.join([str(i['id']) for i in eval_queries]+ [str(i['url']) for i in test_code]))


if __name__ == '__main__':
    CSN_train_path = './raw_dataset/python/train/'
    CSN_test_path = './raw_dataset/python/test/'
    CSN_eval_path = './raw_dataset/python/eval/'
    vocab_path = 'data/python/word_vocab.json'
    model_path = 'main_model/attention_model/main.model'
    save_dir = './processed_dataset/python/attention_model/'
    print("start")
    raw_comments, raw_code = load_CSN(CSN_train_path)
    print('train set size:', len(raw_comments))
    with open(vocab_path,'r') as f:
        word_vocab = json.load(f)
    model = torch.load(model_path)

    queries, code = filtering(raw_comments,raw_code,word_vocab,model)
    
    urls = [co['url'] for co in code]
    with open(save_dir + 'urls.txt', 'w') as f:
        for url in urls:
            f.write(url + '\n')
    print('filtered_size', len(queries))

    extract_features(queries, [co['original_string'] for co in code], [co['func_name'] for co in code], dir_path=save_dir)

    _, test_code = load_CSN(CSN_test_path)
    print(len(test_code))
    extract_test_features(test_code, CSN_eval_path, save_dir)
=======
import nlqf
import os
import os.path as path
import json
import re
import torch
import javalang
import pickle
import numpy as np
import collections
from tables import *
from nltk import stem
from nltk.corpus import stopwords, wordnet
from python_structured import python_query_parse, python_code_parse

P = re.compile(r'([a-z]|\d)([A-Z])')
stemmer = stem.PorterStemmer()
code_stop_words = set(stopwords.words('codeStopWord'))
english_stop_words = set(stopwords.words('codeQueryStopWord'))
ka = re.compile(r'[^a-zA-Z]')


def get_first_sentence(docstring):
    docstring = re.split(r'[.\n\r]', docstring.strip('\n'))[0]
    return docstring


def contain_non_English(line):
    return bool(re.search(r'[^\x00-\xff]', line))


def remove_code_comment(line):
    line = line.replace('\'', '\"')
    line = re.sub(r'\t*"\"\"[\s\S]*?\"\"\"\n+', '', line)
    line = re.sub(r'\t*#+.*?\n+', '', line)
    line = re.sub('\n \n', '\n', line)
    return line


def load_CSN(CSN_path):
    raw_comments = []
    raw_code = []
    for file in os.listdir(CSN_path):
        with open(path.join(CSN_path, file), 'r') as f:
            line_dicts = [json.loads(i) for i in f.readlines()]
            filter_dicts = [d for d in line_dicts if not contain_non_English(str(d))]
            for line in filter_dicts:
                raw_comments.append(get_first_sentence(line['docstring']))
                raw_code.append({
                    'func_name': line['func_name'].split('.')[0],
                    'original_string': remove_code_comment(line['original_string']),
                    'url': line['url']
                })
    return raw_comments, raw_code


def extract_tokens(code_tokens):
    tokens = python_code_parse(code_tokens)
    return ' '.join(tokens)


def extract_docstring(docstring):
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    docstring = python_query_parse(docstring)
    return ' '.join(docstring)


def extract_method_name(function_name):
    function_name = function_name.split('.')[-1]
    sub_name_tokens = remove_underscores(function_name)
    sub_name_tokens = ' '.join(sub_name_tokens)
    return sub_name_tokens


def remove_underscores(function_name):
    cleaned_name = re.sub(r'^_+|_+$', '', function_name)
    cleaned_name = re.sub(r'\d', '', cleaned_name)
    return cleaned_name.lower().split('_')


def generate_vocab(data):
    tokens = ' '.join(data).split(' ')
    count = collections.Counter(tokens)
    tokens_unique = ['<PAD>', '<UNK>'] + [tok for tok, _ in count.most_common(9998)]
    vocab = {token: i for i, token in enumerate(tokens_unique)}
    return vocab


def token2id(data, vocab):
    id_list = []
    for token_str in data:
        tokens = token_str.split()
        ids = []
        for token in tokens:
            ids.append(vocab.get(token, 1))
        if len(ids) == 0:
            ids.append(0)
        id_list.append(ids)
    return id_list


class Features(IsDescription):
    length = Int64Col()
    pos = Int64Col()


def save_features(data, name, dir_path):
    phrases = []
    pos_pointer = 0

    fea_f = open_file(os.path.join(dir_path, name), mode='w',
                      title='Data for deep code search')
    table = fea_f.create_table('/', 'indices', Features, "lalala")
    rows = table.row
    for item in data:
        phrases.extend(item)
        rows['length'] = len(item)
        rows['pos'] = pos_pointer
        pos_pointer += len(item)
        rows.append()
    table.flush()

    arr = fea_f.create_array('/', 'phrases', np.array(phrases))
    arr.flush()
    fea_f.close()


def save_vocab(vocab, name='', dir_path=''):
    os.makedirs(dir_path, exist_ok=True)
    f = open(os.path.join(dir_path, name), 'wb')
    pickle.dump(vocab, f)
    f.close()


def extract_features(doc, code_str, func_name, dir_path):
    docstrings = []
    method_names = []
    codes = []
    for d, c ,f in zip(doc, code_str, func_name):
        docstring = extract_docstring(d)
        cod = extract_tokens(c)
        method_name = extract_method_name(f)
        method_names.append(method_name)
        codes.append(cod)
        docstrings.append(docstring)

    file_name = ['methname', 'code', 'desc']
    vocabs = []
    for i, data_list in enumerate(
            [method_names, codes, docstrings]):
        vocab = generate_vocab(data_list)
        save_vocab(vocab, 'vocab.{}.pkl'.format(file_name[i]), dir_path)
        vocabs.append(vocab)

    for j, data_list in enumerate(
            [method_names, codes, docstrings]):
        train_id_list = token2id(data_list, vocabs[j])

        save_features(train_id_list, '{}.{}.h5'.format('train', file_name[j]),
                      dir_path)


def filtering(raw_comments, raw_code, word_vocab,vae_model):
    comments, idx = nlqf.rule_filter(raw_comments)
    print('after rule filter:', len(comments))
    code = [raw_code[i] for i in idx]
    queries, idx = nlqf.model_filter(comments, word_vocab, vae_model)
    print('after model filter:', len(queries))
    code = [code[i] for i in idx]
    return queries, code


def extract_test_features(codes, query_dir, save_dir):
    eval_queries = []
    with open(path.join(query_dir, 'final_eval.jsonl'), 'r',encoding='utf-8') as f:
        for row in f.readlines():
            eval_queries.append(json.loads(row))
    methname, code, desc = [], [], []
    for i in eval_queries:
        methname.append(extract_method_name(i['method_name']))
        code.append(extract_tokens(i['code']))

    for i in codes:
        methname.append(extract_method_name(i['func_name']))
        code.append(extract_tokens(i['original_string']))
    data_dic = {
        'methname': methname,
        'code': code
    }
    for name,d in data_dic.items():
        with open(os.path.join(save_dir,f'vocab.{name}.pkl'),'rb') as f:
            vocab = pickle.load(f)
        id_list = token2id(d, vocab)
        save_features(id_list, '{}.{}.CSN.h5'.format('use', name), save_dir)
    with open(path.join(save_dir,'codebase_id.txt'), 'w') as f:
        f.write('\n'.join([str(i['id']) for i in eval_queries]+ [str(i['url']) for i in test_code]))


if __name__ == '__main__':
    CSN_train_path = './raw_dataset/python/train/'
    CSN_test_path = './raw_dataset/python/test/'
    CSN_eval_path = './raw_dataset/python/eval/'
    vocab_path = 'data/python/word_vocab.json'
    model_path = 'main_model/attention_model/main.model'
    save_dir = './processed_dataset/python/attention_model/'
    print("start")
    raw_comments, raw_code = load_CSN(CSN_train_path)
    print('train set size:', len(raw_comments))
    with open(vocab_path,'r') as f:
        word_vocab = json.load(f)
    model = torch.load(model_path)

    queries, code = filtering(raw_comments,raw_code,word_vocab,model)
    
    urls = [co['url'] for co in code]
    with open(save_dir + 'urls.txt', 'w') as f:
        for url in urls:
            f.write(url + '\n')
    print('filtered_size', len(queries))

    extract_features(queries, [co['original_string'] for co in code], [co['func_name'] for co in code], dir_path=save_dir)

    _, test_code = load_CSN(CSN_test_path)
    print(len(test_code))
    extract_test_features(test_code, CSN_eval_path, save_dir)
>>>>>>> 13ad5a2... commit
>>>>>>> 7b16988... commitit
