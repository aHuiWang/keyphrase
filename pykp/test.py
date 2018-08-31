# import torch

f = open('/home/wangxingpeng/en/data/kp20k/kp20k_testing.json', 'r')
f1 = open('/home/wangxingpeng/en/data/kp20k/small_kp20k_training.json', 'w')
f2 = open('/home/wangxingpeng/en/data/kp20k/small_kp20k_testing.json', 'w')
line = f.readline()
count = 0
while line:
    count += 1
    if count <= 2000:
        f2.write(line)
    else:
        f1.write(line)
    line = f.readline()
