# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
from evaluate import evaluate_beam_search
import logging
import numpy as np

import config
import utils

import torch
import torch.nn as nn
from torch import cuda

from beam_search import SequenceGenerator
from train import load_data_vocab, init_model, init_optimizer_criterion
from utils import Progbar, plot_learning_curve

import pykp
from pykp.dataloader import KeyphraseDataLoader,KeyphraseDataset,BucketIterator
from pykp.model import Seq2SeqLSTMAttention, Seq2SeqLSTMAttentionCascading

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

def load_data_vocab(opt, load_train=False):

    logging.info("Loading vocab from disk: %s" % (opt.vocab))
    word2id, id2word, vocab = torch.load("../data/small_kp20k/small_kp20k.vocab.pt", 'wb')

    # one2one data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)


    logging.info('======================  Test Dataset  =========================')
    

    opt.word2id = word2id
    opt.id2word = id2word
    opt.vocab = vocab
    
    test_one2many_loader = BucketIterator('../data/small_kp20k/small_kp20k.test.one2many.json',
                                            word2id,id2word,
                                            batch_size=opt.beam_batch,
                                            include_original=True,
                                            mode='test',
                                            repeat=False,
                                            sort=False,
                                            shuffle=False,
                                            length=1875) # 18601
    


    logging.info('#(test data size:  #(one2many pair)=%d, #(batch)=%d' % (len(test_one2many_loader), len(test_one2many_loader)/opt.beam_batch))

    logging.info('#(vocab)=%d' % len(vocab))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return test_one2many_loader, word2id, id2word, vocab


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')

    if opt.cascading_model:
        model = Seq2SeqLSTMAttentionCascading(opt)
    else:
        model = Seq2SeqLSTMAttention(opt)

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        # load the saved the meta-model and override the current one
        model.load_state_dict(torch.load(open('initial.model')))

        if torch.cuda.is_available():
            checkpoint = torch.load(open(opt.train_from, 'rb'))
        else:
            checkpoint = torch.load(
                open(opt.train_from, 'rb'), map_location=lambda storage, loc: storage
            )
        # some compatible problems, keys are started with 'module.'
        checkpoint = dict([(k[7:], v) if k.startswith('module.') else (k, v) for k, v in checkpoint.items()])
        model.load_state_dict(checkpoint)
    else:

        # dump the meta-model
        meta_model_dir = os.path.join(opt.train_from[: opt.train_from.find('.epoch=')], 'initial.model')
        # print(meta_model_dir,'meta_model_dir')
        torch.save(
            model.state_dict(),
            open(meta_model_dir, 'wb')
        )

    if torch.cuda.is_available() and opt.use_gpu:
        model = model.cuda()

    utils.tally_parameters(model)

    return model

def main():
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.preprocess_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    config.predict_opts(parser)
    config.transformer_opts(parser)
    opt = parser.parse_args()

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    # print(opt.gpuid)
    if torch.cuda.is_available() and not opt.gpuid:
        opt.gpuid = 0

    opt.exp = 'predict.' + opt.exp
    if hasattr(opt, 'copy_model') and opt.copy_model:
        opt.exp += '.copy'

    if hasattr(opt, 'bidirectional'):
        if opt.bidirectional:
            opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    logging = config.init_logging(logger_name=None, log_file=opt.exp_path + '/output.log', stdout=True)


    try:
        # opt.train_from = 'model/kp20k.ml.copy.bi-directional.20180823-033739/kp20k.ml.copy.bi-directional.epoch=1.batch=500.total_batch=500.model'
        test_data_loader, word2id, id2word, vocab = load_data_vocab(opt, load_train=False)
        model = init_model(opt)

        generator = SequenceGenerator(model,opt,
                                      eos_id=opt.word2id[pykp.io.EOS_WORD],
                                      beam_size=opt.beam_size,
                                      max_sequence_length=opt.max_sent_length,
                                      )

        model_path = opt.train_from.split('/')[-1]
        _, epoch, batch, total_batch = re.findall('\d+', model_path)

        evaluate_beam_search(generator, test_data_loader, opt, title='predict',
                             save_path=opt.pred_path + '/epoch=%s,batch=%s,total_batch=%s' % (
                             epoch, batch, total_batch))

    except Exception as e:
        logging.exception("message")

if __name__ == '__main__':
    main()
