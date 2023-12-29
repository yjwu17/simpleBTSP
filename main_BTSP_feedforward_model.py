import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

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
fq = 0.005
skips = 1

num_trial = 5
scales = 1

n = int(3.9e4 * scales )
m = int(2.5e4 * scales) # projection from m-dimensional space to n-dimensional space

fp = p = 0.005
p_w = 0.6
disturb = 0.
code_err_tol = 1e-15
prob_poisson = lambda lambdas, k: lambdas ** k / math.factorial(k) * np.exp(-lambdas)
beta_greater = lambda I, k, p: k * math.comb(I, k) * sc.betainc(k, I - k + 1, p) * sc.beta(k, I - k + 1)
beta_less = lambda I, k, p: (I - k) * math.comb(I, k) * sc.betainc(I - k, k + 1, 1 - p) * sc.beta(I - k, k + 1)

"""
generate dataset
"""

record_std = []
record_mean = []
start_time = time.time()
from datetime import date


device = torch.device('cuda')
variable_list1 = np.arange(2e3,3e4+1,2e3).tolist()[::skips]
variable_list2 =  np.arange(0.00,1.01,0.025)[::skips]

acc = 0.
fd = 0.33
data_name = 'Overlaping_masked_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w)  + str(fq)+ '_trial_'+str(num_trial)+'.mat'
def hamming_accuracy(hamming_dist, I, var):
    measure_list = np.arange(I+1)
    acc_list = []
    for d in measure_list:
        acc = (hamming_dist.sum(axis=1) <= d).float().mean().item()
        acc_list.append(var + [d, acc, hamming_dist.mean().item()])
    return acc_list

total_btsp, total_btsp2, total_mfn, total_cfn, total_cfn2, total_mean = [], [],[],[],[],[]
q_num = int(n * fq)
I = int(m * fp)

def avg_pattern_distance(z, M, precison=torch.float16):
    if z.shape[0] != M:
        z = z.T
    assert z.shape[0] == M
    z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
    avg_distance = (z_dist.float().sum()) / M / (M - 1) /z.shape[1]
    return avg_distance.item()

def avg_assembly_overlap(z, M, precison=torch.float32):
    if z.shape[0] != M:
        z = z.T
    assert z.shape[0] == M
    z_dist = z.to(precison) @ z.to(precison).T
    avg_distance = (z_dist.float().sum()-z_dist.diag().sum()) / M / (M - 1)
    return avg_distance.item()

def important_parameter(z1,z2,M):
    z1, z2 = z1.float(), z2.float()
    assemble_size = z1.mean() * n
    z_mask_overlap = ((z1 * z2).sum(0) / (z1.sum(0) + 1)).mean()
    z_pattern_overlap = avg_assembly_overlap(z1, M)
    return [assemble_size.item(), z_mask_overlap.item(),z_pattern_overlap]

 
vth1 = vth2 = int(I * p_w * 0.45) # parmaeter has been optimized by grid search; you can choose any other value for comparison
for idx_var1,var1 in enumerate(variable_list1):
    M = int(var1)
    precison = torch.float16
    start_time = time.time()
    # generate datasets
    arrays = np.arange(0, m - 1)
    data = torch.Tensor(np.random.binomial(n=1, p=p, size=(m, M))).to(device).to(precison)
    """
    training process
    """
    # divide ca1 into two non-overlap subsets
    fq_half = fq / 2
    plateau_winner1 = (torch.rand(n,M).to(device) <= fq_half).to(precison)
    W_mask = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
    import time

    start_time = time.time()
    W = 0
    fq_half = fq / 2
    fq_ltp = fq_half
    fq_ltd = fq_half
    plateau_winner1 = (torch.rand(M, 1, n).to(device) <= fq_ltp).to(precison)
    plateau_winner2 = (torch.rand(M, 1, n).to(device) <= fq_ltd / (1 - fq_ltp)).to(precison)
    plateau_winner2 = plateau_winner2 * (1 - plateau_winner1)  # mutual exclusive
    start_time = time.time()

    # note that the occurance of p1 and p2 is mutual exclusive
    for idx in range(M):
        inpx = data[:, idx].reshape(-1, 1).to(precison)
        W = ((W + inpx @ plateau_winner1[idx, :, :]) >= 1) & (
                (W + inpx @ plateau_winner2[idx, :, :]) <= 1)
    W = W % 2
    W = W_mask * W

    print('finish the generation!', time.time() - start_time)

    for idx_var2, var2 in enumerate(variable_list2):
        fd = var2
        mask_num = int(I * var2)
        start_time = time.time()
        print('\n threshold',vth1,vth2)
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

            """
            recall phase
            """
            query_data = data.clone().to(device).T
            start_time = time.time()
            k = 0
            for i in range(M):
                tmp = query_data[i]
                tmp_idx = torch.where(tmp == 1)[0]
                mask_num = int(len(tmp_idx)*fd)
                random_idx = np.random.choice(len(tmp_idx), size=mask_num, replace=False)
                query_data[i, tmp_idx[random_idx]] = 0
                # np.random.shuffle(mask_data[i])
            query_data = query_data.T
            print(time.time() - start_time)

            torch.cuda.empty_cache()
            z_raw1 = (W.to(precison).T @ data.to(precison) > vth1).t()
            z_raw_rate1 = z_raw1.float().mean().item()
            z_query1 = (W.to(precison).T @ query_data.to(precison) > vth1).t()
            z_query_rate1 = z_query1.float().mean().item()
            z = z_raw1
            
            z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T



            z_raw2 = (W.to(precison).T @ data.to(precison) > vth2).t()
            z_raw_rate2 = z_raw2.float().mean().item()
            z_query2 = (W.to(precison).T @ query_data.to(precison) > vth2).t()
            z_query_rate2 = z_query2.float().mean().item()

            import_param1 = important_parameter(z_raw1,z_query1,M)
            import_param2 = important_parameter(z_raw2,z_query2,M)


            """
            predict
            """
            if trial == 0:
                estimate_hamming1 = mask_sparse_hamming_simple(I, vth1, M, m=m, n=n, fp=fp, fq=fq_half, fd=fd, p_w=p_w)
                estimate_hamming2 = mask_sparse_hamming_simple(I, vth2, M, m=m, n=n, fp=fp, fq=fq_half, fd=fd, p_w=p_w)
                var = [vth1] + [vth2] + [estimate_hamming1] + [estimate_hamming2]
                fr1_pattern_estimate = mask_sparse_hamming_firing_v2(I, vth1, M, m, n, fp, fq_half, 0, p_w)
                fr2_pattern_estimate = mask_sparse_hamming_firing_v2(I, vth2, M, m, n, fp, fq_half, 0, p_w)


            

            hamming_dist1 = abs(z_raw1.float() - z_query1.float())
            btsp_hamming1 = hamming_accuracy(hamming_dist1, I, var)
            btsp_mean1 = hamming_dist1.float().mean().item()

            hamming_dist2 = abs(z_raw2.float() - z_query2.float())
            btsp_hamming2 = hamming_accuracy(hamming_dist2, I, var)
            btsp_mean2 = hamming_dist2.float().mean().item()



            start_time = time.time()

            """
            Record the results
            """

            fr1_pattern = z_raw1.float().mean().item()
            fr2_pattern = z_raw2.float().mean().item()
            w_ratio = W.float().mean().item()/p_w
            ### usueful target: fr1_pattern (assembly size), w_ratio, the reactived size, accordingly, what I need was:
            pred_ratio1 = 1- estimate_hamming1 / (fr1_pattern_estimate + 1e-6)
            pred_ratio2 = 1- estimate_hamming2 / (fr2_pattern_estimate + 1e-6)
            w_ratio_predict = (1-(1-fq*fp/2)**M)
            predict_item =   [fr1_pattern_estimate, fr2_pattern_estimate,
                              w_ratio_predict,
                              pred_ratio1,pred_ratio2  ]
            intereted_item = [fr1_pattern, fr2_pattern,
                              w_ratio,
                              1 -btsp_mean1 /(fr1_pattern + 1e-6), 1 - btsp_mean2 / (fr2_pattern + 1e-6)]
            

            pattern_dist = import_param1 + import_param2 + predict_item + intereted_item


            record.append([var1, var2,
                           vth1, vth2,
                           btsp_mean1, btsp_mean2, ] + pattern_dist)
            record_btsp.append(btsp_hamming1)
            record_btsp2.append(btsp_hamming2)
            del  z_query_rate1,z_query_rate2;   torch.cuda.empty_cache()
            print('stored patterns', M, 'masking ratio', var2, 'firing rate', z_raw_rate1, 'HD ratio',
                  btsp_mean1 / pattern_dist[-4], btsp_mean2 / pattern_dist[-3], )
 



