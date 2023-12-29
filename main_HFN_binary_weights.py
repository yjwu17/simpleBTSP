import os

node_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = node_id
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from binarize_hopfield_models import Binarize_Hopfield, Hopfield
import copy
# from utils import *
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
num_trial = 1
scales = 1
n = int(3.9e4 * scales)
m = int(2.5e4 * scales)  # projection from m-dimensional space to n-dimensional space
fq = 0.005 / 2
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
# variable_list1 = [1e1, 5e1, 1e2,2e2, 5e2, 1e3, 2.5e3, 5e3, 7.5e3, 1e4,  1.5e4, 2.9e4]
variable_list1 = np.arange(1e1, 3e4, 2e3)
variable_list2 = np.arange(0.00, 1.01, 0.01)
vth_list = [-1, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05]  # grid search for the threshold of binarization HFN

acc = 0.
cdf_vth = fp * 0.2
cdf_fp = cdf_vth  # define below
num_iter_hfn = 100

mode = 'median'  # median, 0
data_name = 'HFN_binary_weight_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w) + '_vth_' + str(
	cdf_vth) + mode + '_v' + node_id + '.mat'


def hamming_accuracy(hamming_dist, I, var):
	measure_list = np.arange(I + 1)
	acc_list = []
	for d in measure_list:
		acc = (hamming_dist.sum(axis=1) <= d).float().mean().item()
		acc_list.append(var + [d, acc, hamming_dist.mean().item()])
	return acc_list


total_btsp, total_btsp2, total_mfn, total_cfn, total_cfn2, total_mean = [], [], [], [], [], []
I = int(m * fp)

w_range = 8
weight_matrix = torch.randint(-w_range, w_range + 1, (1, m)).cuda().float()
vth_threshold = 0
linear_classifier = lambda x: (weight_matrix @ x.float() > vth_threshold)


def avg_pattern_distance(z, M, precison=torch.float16):
	if z.shape[0] != M:
		z = z.T
	assert z.shape[0] == M
	z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
	avg_distance = (z_dist.float().sum()) / M / (M - 1) / z.shape[1]
	return avg_distance.item() + 1e-40


precison = torch.float16
for idx_var1, var1 in enumerate(variable_list1):
	M = int(var1)
	
	for idx_var2, var2 in enumerate(variable_list2):
		mask_num = int(I * var2)
		start_time = time.time()
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
			data = torch.Tensor(np.random.binomial(n=1, p=p, size=(m, M))).to(device).int()
			"""
			training process
			"""
			CFN = Binarize_Hopfield(raw_data=data, m=m, p=fp, precision=precison, p_w=1, binarize_threshold=mode)
			cdf_fp = CFN.W.float().mean()
			"""
			calculate the recall rate
			"""
			query_data = data.clone().to(device).T
			start_time = time.time()
			for i in range(M):
				tmp = query_data[i]
				
				tmp_idx0 = torch.where(tmp == 0)[0]
				tmp_idx1 = torch.where(tmp == 1)[0]
				
				mask_num = int(len(tmp_idx1) * var2)
				random_idx1 = np.random.choice(len(tmp_idx1), size=mask_num, replace=False)
				query_data[i, tmp_idx1[random_idx1]] = 0
			
			query_data = query_data.T
			print(time.time() - start_time)
			z_raw_rate1 = z_raw_rate2 = 0.5
			z_query1 = 0.5
			z_query_rate2 = 0.5
			torch.cuda.empty_cache()
			"""
			predict
			"""
			if trial == 0:
				estimate_hamming1 = 0.5
				estimate_hamming2 = 0.5
				var = [1] + [1] + [estimate_hamming1] + [estimate_hamming2]
				fr1_pattern = 0.5
				fr2_pattern = 0.5
			
			ratio_inp_record = []
			ratio_out_record = []
			ratio_fr_record = []
			for vth in vth_list:
				if vth == -1:
					cdf_vth = CFN.W.mean()
				else:
					cdf_vth = vth
				cfn_query1 = CFN.update(num_updates=num_iter_hfn, query_data=query_data, precision=precison,
				                        threshold=cdf_vth)
				cfn_pattern1 = CFN.update(num_updates=num_iter_hfn, query_data=data, precision=precison,
				                          threshold=cdf_vth)
				
				data_distance = (abs(query_data.float() - data.float()).mean().item() + 1e-4)
				hfn_c1 = abs(cfn_query1.float() - data.float()).mean().item() / data_distance
				hdf_r1 = abs(cfn_query1.float() - cfn_pattern1.float()).mean().item() / data_distance
				ratio_inp_record.append(hfn_c1)
				ratio_out_record.append(hdf_r1)
				ratio_fr_record.append(cfn_pattern1.mean().item())
			
			items = [var1, var2] + ratio_inp_record + ratio_out_record
			torch.cuda.empty_cache()
			record.append(items)
			
			print('var', var1, var2, 'firing rate', cfn_pattern1.mean().item(), cfn_query1.mean().item(),
			      '\n\n input_completion', ratio_inp_record, 'projection', ratio_out_record, 'fr', ratio_fr_record)
			del data;
			torch.cuda.empty_cache()
 