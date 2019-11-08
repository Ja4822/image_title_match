import os
import cv2
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

RESULTS_PATH = '../data/text_vectors.csv'
TITLES_CSV_FILE = '../data/sorted_titles.csv'

def plot_cos_sim(matrix, img_idx, category):
    #gs = gridspec.GridSpec(10, 10)
    m, n = matrix.shape
    for i in range(m):
        row = np.array(matrix.iloc[i,:])
        plt.figure(figsize=(13.6,10.2))
        plt.bar(range(len(row)), row)
        plt.ylim(0,1)
        plt.grid(axis='y')
        # plt.savefig('../data/cos_sim_results/%s/%d.png'%(category, img_idx[i]))

if __name__ == "__main__":
    
    print('=============== LOAD DATASET ===============')
    titles = pd.read_csv(TITLES_CSV_FILE)
    text_vectors = pd.read_csv(RESULTS_PATH)
    num_rows, num_cols = text_vectors.shape

    print('======= SEPERATE SAMPLES WITH TRUMP ========')
    trump_idx_list = []
    no_trump_idx_list = []
    for i in range(num_cols):
        row = str(titles.iloc[i])
        row = row.lower()
        if 'trump' in row:
            trump_idx_list.append(i)
        else:
            no_trump_idx_list.append(i)
    print(trump_idx_list)
    print('[INFO] sample    with \'trump\' num = %d'%(len(trump_idx_list)))
    print('[INFO] sample without \'trump\' num = %d'%(len(no_trump_idx_list)))

    print('=============== PLOT COS SIM ===============')
    # plot_cos_sim(in_trump_matrix, trump_idx_list, 'in_trump')
    # plot_cos_sim(no_trump_matrix, no_trump_idx_list, 'no_trump')
    # plot_cos_sim(between_trump_matrix, trump_idx_list, 'between_trump')

    num = 0
    err_list = []
    for i in range(len(trump_idx_list)):
        row = np.array(text_vectors.iloc[trump_idx_list[i], :])
        tmp_list = []
        for j in range(len(trump_idx_list)):
            tmp = round(row[trump_idx_list[j]], 3)
            tmp_list.append(tmp)
        trump_mean = (np.sum(tmp_list)-1)/(len(trump_idx_list)-1)
        for j in range(len(no_trump_idx_list)):
            tmp = round(row[no_trump_idx_list[j]], 3)
            tmp_list.append(tmp)
        no_trump_mean = np.mean(tmp_list)
        if trump_mean > no_trump_mean:
            num += 1
        err_list.append(trump_mean - no_trump_mean)
        print('text %d: trump_mean = %.3f, no_trump_mean = %.3f'%(trump_idx_list[i], trump_mean, no_trump_mean))
    print('same category percentage = %.3f'%(num/len(trump_idx_list)))
    print('distribution of trump - no_trump = (%.3f +- %.3f)'%(np.mean(err_list), np.std(err_list)))
    # figsize=(13.6,10.2)
    # plt.figure()
    # plt.bar(range(len(mean_list)), mean_list)
    # plt.xlabel('img index')
    # plt.ylabel('cos sim')
    # plt.ylim(0,0.4)
    # plt.grid(axis='y')
    # plt.show()

