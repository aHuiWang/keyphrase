# -*- coding:utf-8 -*-
__author__ = 'wanghui'
__date__ = '2018/8/18 11:05'

# import re
#
# path = "/home/wangxingpeng/en/model/kp20k.ml.copy.uni-directional.20180817-192718/" \
#        "kp20k.ml.copy.uni-directional.epoch=1.batch=100.total_batch=100.model"
# string = path.split('/')[-1]
# _, epoch, batch, total_batch = re.findall(r"\d+",string)
#
# print(epoch, batch, total_batch)

import os
import time

mem_pred_shell = "python predict.py -train_from {train_from} -timemark small_kp20k_mem -beam_size 20 -beam_batch 1 -decoder_type 'Memory Network' -report_file mem.res.txt"
copy_pred_shell = "python predict.py -train_from {train_from} -timemark small_kp20k_copy -beam_size 20 -beam_batch 1 -decoder_type CopyRNN -report_file copy.res.txt"
# -encoder_type BiGRU -decoder_type 'Memory Network' -report_file mem.res.txt
# -encoder_type Attention -decoder_type CopyRNN
# -encoder_type BiGRU -decoder_type CopyRNN -report_file copy.res.txt

# -word_vec_size {embedding_size} -rnn_size {rnn_size}

mem_train_shell = "python train.py -timemark small_kp20k_mem -batch_size 200 -save_model_every 300 -decoder_type 'Memory Network'"
copy_train_shell = "python train.py -timemark small_kp20k_copy -batch_size 200 -save_model_every 300 -decoder_type CopyRNN"
# -copy_attention defalut True -decoder_type Memory Network
size_list = ['100', '200']
mem_model_path = "/home/wangxingpeng/en/keyphrase/model/kp20k.ml.copy.bi-directional.small_kp20k_mem/kp20k.ml.copy.bi-directional.epoch=9.batch=404.total_batch=3900.model"
copy_model_path = "/home/wangxingpeng/en/keyphrase/model/kp20k.ml.copy.bi-directional.small_kp20k_copy/kp20k.ml.copy.bi-directional.epoch=9.batch=404.total_batch=3900.model"
# for i in size_list:
# os.system(train_shell)
os.system(mem_train_shell)
os.system(copy_train_shell)
os.system(mem_pred_shell.format(train_from=mem_model_path))
os.system(copy_pred_shell.format(train_from=copy_model_path))

# print("Sleep!")
# time.sleep(2*60*60)
#
# nohup_shell = "nohup python -u crawl_activity.py > {c}.out 2>&1 &"
# for i in range(1, 26):
#     print(i, ' Yes!')
#     os.system(nohup_shell.format(c=str(i)))




