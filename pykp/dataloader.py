# -*- coding: utf-8 -*-
"""
Large chunk borrowed from PyTorch DataLoader
"""

import os
import json
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torch.autograd import Variable
import collections
import re
import sys
import traceback
import threading
import itertools

if sys.version_info[0] == 2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DIGIT = '<digit>'


def KeyphraseDataset(datapath, word2id, id2word,include_original=False):
    
    def preocess_example(e):
        keys = ['src', 'trg', 'trg_copy', 'src_oov', 'oov_dict', 'oov_list','query']
        if include_original:
            keys += ['src_str','trg_str']

        filtered_example = {}
        for k in keys:
            filtered_example[k] = e[k]
        if 'oov_list' in filtered_example:
            if type == 'one2one':
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            elif type == 'one2many':
                filtered_example['oov_number'] = [len(oov) for oov in filtered_example['oov_list']]
        filtered_example['src_len'] = len(filtered_example['src'])
        return filtered_example
    
    f = open(datapath)
    for line in f:
        sample = json.loads(line.strip())
        yield preocess_example(sample)
    
class BucketIterator(object):
    """docstring for BucketIterator"""
    def __init__(self,datapath,word2id,id2word,
            batch_size=10,sort=False,
            shuffle=None,repeat=False,
            mode=None,
            include_original=False,
            length=None,Data_type=None):

        super(BucketIterator, self).__init__()#dataset=dataset,sort=False,shuffle=False,
                                            #batch_size=batch_size,repeat=repeat)
        self.datapath = datapath
        
        self.word2id = word2id
        self.id2word = id2word
        self.batch_size = batch_size

        self.include_original = include_original        
        self.pad_id = word2id[PAD_WORD]
        
        self.repeat = repeat
        self.sort = sort
        self.length=length
        self.mode = mode
        self.Data_Set=Data_type
    def __len__(self):
        return self.length

    def init_batches(self):
        # self.dataset = KeyphraseDataset(self.datapath,self.word2id,self.id2word,self.include_original)
        self.dataset = self.Data_Set(self.datapath,self.word2id,self.id2word,self.include_original)
        
        if self.mode == 'keyword':
            self.batches = self.pool(self.dataset, self.batch_size,self.collate_fn_one2one) 
        elif self.mode =='keyphrase':
            self.batches = self.pool(self.dataset, self.batch_size,self.collate_fn_one2many)
        else:
            self.batches = self.pool(self.dataset,self.batch_size,self.process_batch)
    
    def batch(self,data, batch_size):
    
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = len(minibatch)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            
        if minibatch:
            yield minibatch
    
    def pool(self,data, batch_size,collate_fn=None):

        for p in self.batch(data, batch_size * 100):
            p_batch =  self.batch(p, batch_size)
            
            for b in p_batch:
                yield collate_fn(b)
           
    def collate_fn_one2one(self, batches):
        '''
        Puts each data field into a tensor with outer dimension batch size"
        '''
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        trg = [[self.word2id[BOS_WORD]] + b['trg'] + [self.word2id[EOS_WORD]] for b in batches]

        # target_for_loss: input to criterion, if it's copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_target = [b['trg'] + [self.word2id[EOS_WORD]] for b in batches]
        trg_copy_target = [b['trg_copy'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (unk words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        
        oov_lists = [b['oov_list'] for b in batches]
        query_lists = [b['query'] for b in batches]

        if self.include_original:
            src_str = [b['src_str'] for b in batches]
            trg_str = [b['trg_str'] for b in batches]

        if self.sort:
            src_len_order = np.argsort([len(x) for x in src])[::-1]
            src = [src[i] for i in src_len_order]
            src_oov = [src_oov[i] for i in src_len_order]
            trg = [trg[i] for i in src_len_order]
            trg_target = [trg_target[i] for i in src_len_order]
            trg_copy_target = [trg_copy_target[i] for i in src_len_order]
            oov_lists = [oov_lists[i] for i in src_len_order]
            query_lists = [query_lists[i] for i in src_len_order]

            if self.include_original:
                src_str = [src_str[i] for i in src_len_order]
                trg_str = [trg_str[i] for i in src_len_order]

        src, src_len, src_mask = self._pad(src)
        trg, _, _ = self._pad(trg)
        trg_target, _, _ = self._pad(trg_target)
        trg_copy_target, _, _ = self._pad(trg_copy_target)
        src_oov, _, _ = self._pad(src_oov)
        query_lists,_,_ = self._pad(query_lists)       
             
        if self.include_original:
            return src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists,query_lists,src_str,trg_str

        return src, src_len, trg, trg_target, trg_copy_target, src_oov, oov_lists,query_lists

    def collate_fn_one2many(self, batches):
        # source with oov words replaced by <unk>
        src = [[self.word2id[BOS_WORD]] + b['src'] + [self.word2id[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [[self.word2id[BOS_WORD]] + b['src_oov'] + [self.word2id[EOS_WORD]] for b in batches]
        # target_input: input to decoder, starts with BOS and oovs are replaced with <unk>
        

        trg = [[[self.word2id[BOS_WORD]] + t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]

        # target_for_loss: input to criterion
        trg_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg']] for b in batches]
        # target for copy model, oovs are replaced with temporary idx, e.g. 50000, 50001 etc.)
        trg_copy_target = [[t + [self.word2id[EOS_WORD]] for t in b['trg_copy']] for b in batches]
        oov_lists = [b['oov_list'] for b in batches]
        query_lists = [b['query'] for b in batches]

        # for training, the trg_copy_target_o2o and trg_copy_target_o2m is the final target (no way to uncover really unseen words). for evaluation, the trg_str is the final target.
        if self.include_original:
            src_str = [b['src_str'] for b in batches]
            trg_str = [b['trg_str'] for b in batches]

        # sort all the sequences in the order of source lengths, to meet the requirement of pack_padded_sequence
        src_len_order = np.argsort([len(s) for s in src])[::-1]
        src = [src[i] for i in src_len_order]
        src_oov = [src_oov[i] for i in src_len_order]
        trg = [trg[i] for i in src_len_order]
        trg_target = [trg_target[i] for i in src_len_order]
        trg_copy_target = [trg_copy_target[i] for i in src_len_order]
        oov_lists = [oov_lists[i] for i in src_len_order]
        query_lists = [query_lists[i] for i in src_len_order]

        if self.include_original:
            src_str = [src_str[i] for i in src_len_order]
            trg_str = [trg_str[i] for i in src_len_order]

        # pad the one2many variables
        src_o2m, src_o2m_len, _ = self._pad(src)
        trg_o2m = trg
        src_oov_o2m, _, _ = self._pad(src_oov)
        # trg_target_o2m, _, _      = self._pad(trg_target)
        trg_copy_target_o2m = trg_copy_target
        oov_lists_o2m = oov_lists
        # query_lists,_,_ = self._pad(query_lists)
        
        query_lists = torch.LongTensor(query_lists)        
        # unfold the one2many pairs and pad the one2one variables
        src_o2o, src_o2o_len, _ = self._pad(list(itertools.chain(*[[src[idx]] * len(t) for idx, t in enumerate(trg)])))
        src_oov_o2o, _, _ = self._pad(list(itertools.chain(*[[src_oov[idx]] * len(t) for idx, t in enumerate(trg)])))
        trg_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg])))
        trg_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_target])))
        trg_copy_target_o2o, _, _ = self._pad(list(itertools.chain(*[t for t in trg_copy_target])))
        oov_lists_o2o = list(itertools.chain(*[[oov_lists[idx]] * len(t) for idx, t in enumerate(trg)]))

        assert (len(src) == len(src_o2m) == len(src_oov_o2m) == len(trg_copy_target_o2m) == len(oov_lists_o2m))
        assert (sum([len(t) for t in trg]) == len(src_o2o) == len(src_oov_o2o) == len(trg_copy_target_o2o) == len(oov_lists_o2o))
        assert (src_o2m.size() == src_oov_o2m.size())
        assert (src_o2o.size() == src_oov_o2o.size())
        assert ([trg_o2o.size(0), trg_o2o.size(1) - 1] == list(trg_target_o2o.size()) == list(trg_copy_target_o2o.size()))

        # return two tuples, 1st for one2many and 2nd for one2one (src, src_oov, trg, trg_target, trg_copy_target, oov_lists)
        if self.include_original:
            return src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m, query_lists,src_str, trg_str
        else:
            return src_o2m, src_o2m_len, trg_o2m, None, trg_copy_target_o2m, src_oov_o2m, oov_lists_o2m,query_lists

    def _pad(self,x_raw,pad_id=0):
        x_raw = np.asarray(x_raw)
        x_lens = [len(x_) for x_ in x_raw]
        max_length = max(x_lens)  # (deprecated) + 1 to ensure at least one padding appears in the end
        # x_lens = [x_len + 1 for x_len in x_lens]
        x = np.array([np.concatenate((x_, [pad_id] * (max_length - len(x_)))) for x_ in x_raw])
        x = Variable(torch.stack([torch.from_numpy(x_) for x_ in x], 0)).type('torch.LongTensor')
        x_mask = np.array([[1] * x_len + [0] * (max_length - x_len) for x_len in x_lens])
        x_mask = Variable(torch.stack([torch.from_numpy(m_) for m_ in x_mask], 0))

        assert x.size(1) == max_length

        return x, x_lens, x_mask
    
    # def process_batch(self,batch):
    #     print('fuck')
    #     pass 

    def __iter__(self):
        while True:
            self.init_batches()
            for idx, minibatch in enumerate(self.batches):
                yield minibatch
            if not self.repeat:
                return

if __name__ == '__main__':
    word2id,id2word,vocab = torch.load('../data/AAAI/kp20k.vocab.pt')
    
    test_one2many_loader = BucketIterator('../data/AAAI/kp20k.test.one2many.json',word2id,id2word,
                                            batch_size=opt.beam_batch,
                                            include_original=True,
                                            mode='test',
                                            repeat=False,
                                            sort=False,
                                            shuffle=False,
                                            length=18601)


    for idx, batch in enumerate(test_one2many_loader):
        if idx > 1000:
            print(idx)
    print(done)