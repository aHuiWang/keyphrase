# -*- coding: utf-8 -*-
# import torch
import json

def split_kp20k_test(filename, train_file, test_file):

    f = open(filename, 'r')
    f1 = open(train_file, 'w')
    f2 = open(test_file, 'w')
    line = f.readline()
    count = 0
    while line:
        count += 1
        if count <= 2000:
            f2.write(line)
        else:
            f1.write(line)
        line = f.readline()

def process_pred():
    file1 = 'copy.res.txt'
    file2 = 'mem.res.txt'
    fr1 = open(file1, 'r')
    fw1 = open('copyrnn_all.txt', 'w')
    fr2 = open(file2, 'r')
    fw2 = open('memery_all.txt', 'w')
    line1 = fr1.readline().strip()
    line2 = fr2.readline()
    count = 0
    while line1:
        data1 = json.loads(line1)
        data2 = json.loads(line2)
        count += 1
        data1['src_str'] = ' '.join(data1['src_str'])
        data1['trg_str_seqs'] = ';'.join([' '.join(w) for w in data1['trg_str_seqs']])
        data1['pred'] = ';'.join([' '.join(w) for w in data1['pred']])
        fw1.write(json.dumps(data1)+'\n')
        data2['src_str'] = ' '.join(data2['src_str'])
        data2['trg_str_seqs'] = ';'.join([' '.join(w) for w in data2['trg_str_seqs']])
        data2['pred'] = ';'.join([' '.join(w) for w in data2['pred']])
        fw2.write(json.dumps(data2)+'\n')
        line1 = fr1.readline().strip()
        line2 = fr2.readline().strip()

def count_PRF(src, trg_str, preds_str, topK):
    match = 0
    first_num = 0 # 统计出错的词 第一个单词就错误的个数
    appear_num = 0 # 没有成功召回的词 第一个词在文中的个数
    trg = trg_str.split(';')
    trg_copy = trg[:] # trg_copy 留存没有召回的词
    preds = preds_str.split(';')


    for pred in preds[:topK]:
        if pred in trg:
            trg_copy.remove(pred)
            match += 1
        else:# 不对的词 第一个单词错误的比例
            first_word = pred.split(' ')[0]
            right_flag = False # 默认第一个词不对
            for i in trg:
                if i.split(' ')[0] == first_word:
                    right_flag = True
                    break
            if not right_flag:
                first_num += 1

    for i in trg_copy:
        if i.split(' ')[0] in src:
            appear_num += 1
    try:
        first_rate = float(first_num)/(topK-match)
    except:
        first_rate = 0.0
    try:
        appear_rate = float(appear_num)/len(trg_copy)
    except:
        appear_rate = 0.0


    P = float(match)/topK
    R = float(match)/len(trg)
    try:
        F = 2*P*R/(P+R)
    except:
        F = 0.0

    return match, P, R, F, first_rate, appear_rate


def compare_pred(topK):
    file1 = 'copyrnn_all.txt'
    file2 = 'memery_all.txt'

    fr1 = open(file1, 'r')
    fr2 = open(file2, 'r')
    fw = open('compare_'+str(topK)+'.txt', 'w')
    line1 = fr1.readline().strip()
    line2 = fr2.readline().strip()
    count = 0
    copyGmem = []
    copyEmem = []
    copyLmem = []
    F1_total = 0.0
    F2_total = 0.0
    first_total1 = 0.0
    first_total2 = 0.0
    appear_total1 = 0.0
    appear_total2 = 0.0
    while line1:
        data1 = json.loads(line1)
        data2 = json.loads(line2)
        count += 1
        num1, P1, R1, F1, first1, appear1 \
            = count_PRF(data1['src_str'], data1['trg_str_seqs'], data1['pred'], topK)
        F1_total += F1
        first_total1 += first1
        appear_total1 += appear1
        newline = str(num1) + '\t' + str(P1) + '\t'+ str(R1) +'\t' + str(F1) + '\t'
        num2, P2, R2, F2, first2, appear2 \
            = count_PRF(data2['src_str'], data2['trg_str_seqs'], data2['pred'], topK)
        F2_total += F2
        first_total2 += first2
        appear_total2 += appear2
        newline += str(num2) + '\t' + str(P2) + '\t' + str(R2) + '\t' + str(F2) + '\n'
        fw.write(newline)
        if F1 > F2:
            copyGmem.append(str(count))
        elif F1 == F2:
            copyEmem.append(str(count))
        else:
            copyLmem.append(str(count))
        line1 = fr1.readline().strip()
        line2 = fr2.readline().strip()
    print(topK)
    print('CopyRNN', 'F1', F1_total/count, '错误结果第一个词不对',first_total1/count,
          '未成功召回第一个词在文中', appear_total1/count)
    print('Memery', 'F1', F2_total/count, '错误结果第一个词不对', first_total2/count,
          '未成功召回第一个词在文中', appear_total2/count)
    fw.write('F1 Copy > Mem Num: '+ str(len(copyGmem)) + '\nLineNum:'+ ' '.join(copyGmem)+'\n')
    fw.write('F1 Copy = Mem Num: '+ str(len(copyEmem)) + '\nLineNum:'+ ' '.join(copyEmem)+'\n')
    fw.write('F1 Copy < Mem Num: '+ str(len(copyLmem)) + '\nLineNum:'+ ' '.join(copyLmem)+'\n')


process_pred()
compare_pred(5)
compare_pred(10)
# split_kp20k_test('kp20k_testing_query_10.json', 'small_train_10.json', 'small_test_10.json')
# split_kp20k_test('kp20k_testing_query_5.json', 'small_train_5.json', 'small_test_5.json')
# def trans_pt2json(filename):
#     json_file_ = filename.split('.')
#     json_file_[-1] = 'json'
#     json_file = '.'.join(json_file_)
#     data = torch.load(filename)
#     fw = open(json_file, 'w')
#     for i in data:
#         fw.write(json.dumps(i)+'\n')
#
#
# file_list = ['small_kp20k.test.one2many.pt', 'small_kp20k.train.one2one.pt']
# for i in file_list:
#     trans_pt2json(i)


