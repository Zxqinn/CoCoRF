import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
from utils import PAD_ID

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, max_name_len,
                 f_code, max_code_len, f_descs=None, max_desc_len=None):
        self.max_name_len=max_name_len
        self.max_code_len=max_code_len
        self.max_desc_len=max_desc_len
        # 1. Initialize file path or list of file names.
        """read training data(list of int arrays) from a hdf5 file"""
        self.training=False
        print("loading data...")

        table_name = tables.open_file(data_dir+f_name)
        self.names = table_name.get_node('/phrases')[:].astype(np.long)
        self.idx_names = table_name.get_node('/indices')[:]
        table_code = tables.open_file(data_dir+f_code)
        self.code = table_code.get_node('/phrases')[:].astype(np.long)
        self.idx_code = table_code.get_node('/indices')[:]

        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
        
        assert self.idx_names.shape[0] == self.idx_code.shape[0]
        if f_descs is not None:
            assert self.idx_names.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.idx_names.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq=np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq=seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):          
        len, pos = self.idx_names[offset]['length'], self.idx_names[offset]['pos']
        name_len=min(int(len),self.max_name_len) 
        name = self.names[pos: pos+name_len]
        name = self.pad_seq(name, self.max_name_len)

        len, pos = self.idx_code[offset]['length'], self.idx_code[offset]['pos']
        code_len = min(int(len), self.max_code_len)
        code = self.code[pos:pos + code_len]
        code = self.pad_seq(code, self.max_code_len)

        if self.training:
            len, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
            good_desc_len = min(int(len), self.max_desc_len)
            good_desc = self.descs[pos:pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            rand_offset=random.randint(0, self.data_len-1)
            len, pos = self.idx_descs[rand_offset]['length'], self.idx_descs[rand_offset]['pos']
            bad_desc_len=min(int(len), self.max_desc_len)
            bad_desc = self.descs[pos:pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

            return name, name_len, code, code_len, good_desc, good_desc_len, bad_desc, bad_desc_len
        return name, name_len, code, code_len,
        
    def __len__(self):
        return self.data_len
    
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        lens = [len(d) for d in datasets]
        assert len(np.unique(lens)) == 1
        
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
