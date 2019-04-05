import matplotlib.pyplot as plt
import os
import numpy as np

acc_list = []

file_path = '/disk2/lzq/results/MERL/split_2/resnet50_100epoch.txt'
with open(file_path, 'r') as f:
    for line in f.readlines():
        acc_str = line.split(',')[1]
        acc_list.append(float(acc_str.split('=')[1].split('\n')[0])*100)


def plot_history(acc):
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(np.arange(len(acc)), acc)
    plt.legend(['Training'])
    plt.show()

plot_history(acc_list)
