import os
node_id = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = node_id
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from continuous_hopfield_models import Hopfield
import copy
from utils import *
import time
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import kstest, anderson, shapiro
import numpy as np
import matplotlib

import scipy.special as sc

import math
from theory_prediction_function import *
"""
parameter setting
"""
num_trial = 3
scales = 1
n = int(3.9e4 * scales)
m = int(2.5e4 * scales) # projection from m-dimensional space to n-dimensional space
fq = 0.005/2
fp = p = 0.005
p_w = 0.6
"""
generate dataset
"""
record_std = []
record_mean = []
start_time = time.time()
from datetime import date
device = torch.device('cuda')
variable_list1 = np.arange(1e3,3e4+1,1e3).tolist()
variable_list2 =  np.arange(0.00,0.41,0.02)

cdf_vth = fp * 0.4
cdf_vth2 = fp * 0.3
num_iter_hfn = 100
data_name = 'HFN_mask_comm_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w) + '_vth1' + str(
    cdf_vth) + '_vth2_'+str(cdf_vth2) + '_iter_' + str(num_iter_hfn) + '_v' + node_id +'.mat'
def hamming_accuracy(hamming_dist, I, var):
    measure_list = np.arange(I+1)
    acc_list = []
    for d in measure_list:
        acc = (hamming_dist.sum(axis=1) <= d).float().mean().item()
        acc_list.append(var + [d, acc, hamming_dist.mean().item()])
    return acc_list

total_btsp, total_btsp2, total_mfn, total_cfn, total_cfn2, total_mean = [], [],[],[],[],[]
I = int(m * fp)
def avg_pattern_distance(z, M, precison=torch.float16):
    if z.shape[0] != M:
        z = z.T
    assert z.shape[0] == M
    z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
    avg_distance = (z_dist.float().sum()) / M / (M - 1) /z.shape[1]
    return avg_distance.item() + 1e-7

precison = torch.float16
for idx_var1,var1 in enumerate(variable_list1):
    M = int(var1)
    vth1 = int(I*p_w*0.5)
    vth2 = vth1

    print('\n threshold', vth1, vth2)
    mask_num = int(I * 0.33)
    for idx_var2, var2 in enumerate(variable_list2):

        start_time = time.time()
        common_ratio = var2
        comm_num = int(I * common_ratio)
        p_rest = (I - comm_num) / (m - comm_num)

        """
        Experiments
        """
        record_btsp = []
        record_btsp2 = []
        record = []
        record_cdf = []
        record_cdf2 = []
        record_mdf = []
        for trial in range(num_trial):
            start_time = time.time()
            # generate datasets
            arrays = np.arange(0, m - 1)
            data = torch.Tensor(np.random.binomial(n=1, p=p_rest, size=(m, M))).to(device).int()
            data[:comm_num, :] = 1
            # data = torch.Tensor(np.random.binomial(n=1, p=p, size=(m, M))).to(device).int()
            """
            training process
            """
            CFN = Hopfield(raw_data=data, m=m, p=fp,  precision=precison, p_w = p_w)
            """
            calculate the recall rate
            """
            query_data = data.clone().to(device).T
            start_time = time.time()
            k = 0
            for i in range(M):
                tmp = query_data[i]
                tmp_idx = torch.where(tmp == 1)[0]
                if len(tmp_idx) > mask_num:
                    random_idx = np.random.choice(len(tmp_idx), size=mask_num, replace=False)
                    query_data[i, tmp_idx[random_idx]] = 0
                else:
                    query_data[i] = 0
                    k += 1
                # np.random.shuffle(mask_data[i])
            query_data = query_data.T
            print(time.time() - start_time)

            torch.cuda.empty_cache()
            """
            predict
            """
            var = [0]

            cfn_query1 = CFN.update(num_updates=num_iter_hfn, query_data=query_data, precision=precison, threshold=cdf_vth)
            cfn_pattern1 = CFN.update(num_updates=num_iter_hfn, query_data=data, precision=precison, threshold=cdf_vth)
            cfn_query2 = CFN.update(num_updates=num_iter_hfn, query_data=query_data, precision=precison, threshold=cdf_vth2)
            cfn_pattern2 = CFN.update(num_updates=num_iter_hfn, query_data=data, precision=precison, threshold=cdf_vth2)

            torch.cuda.empty_cache()
            hamming_dist3 = abs(cfn_query1.float() - data.float())
            cfn_hamming = hamming_accuracy(hamming_dist3.T, I, var)
            cfn_mean = hamming_dist3.float().mean().item()

            hamming_dist4 = abs(cfn_query1.float() - cfn_pattern1.float())
            cfn_hamming2 = hamming_accuracy(hamming_dist4.T, I, var)
            cfn_mean2 = hamming_dist4.float().mean().item()
            
            hamming_dist1 = abs(cfn_query2.float() - data.float())
            btsp_hamming1 = hamming_accuracy(hamming_dist1, I, var)
            btsp_mean1 = hamming_dist1.float().mean().item()

            hamming_dist2 = abs(cfn_query2.float() - cfn_pattern2.float())
            btsp_hamming2 = hamming_accuracy(hamming_dist2, I, var)
            btsp_mean2 = hamming_dist2.float().mean().item()

            torch.cuda.empty_cache()
            start_time = time.time()

            """
            Record the results
            """

            hfn_xaa = hamming_dist3.float().mean().item()
            hfn_zaa = hamming_dist4.float().mean().item()
            hfn_ab = avg_pattern_distance(cfn_pattern2,M)
            ratios = hfn_zaa/hfn_ab
            hdf_r1 = btsp_mean2/avg_pattern_distance(cfn_pattern1,M)
            hdf_r2 = hfn_xaa/hfn_ab
            
            print('\n\n HFN M',var1,'comm  fraction',var2,' zaa',hfn_xaa*int(2.5e4),'HFN r1',hdf_r1,'HFN r2',hdf_r2,ratios)



   


