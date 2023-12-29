import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from hopfield_models import Hopfield, ModernHopfield
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
scales = 1.
n = int(3.9e4 * scales)
m = int(2.5e4 * scales)  # projection from m-dimensional space to n-dimensional space
fq = 0.005 / 2
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
variable_list1 = np.arange(1e3, 3e4 + 1, 2e3).tolist()
variable_list2 = np.arange(0.00, 0.91, 0.02)
acc = 0.
fd = 0.33
data_name = 'fig3_BTSP_RP_masked_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w) + str(fq) + '_trial_' + str(
	num_trial) + '.mat'


def hamming_accuracy(hamming_dist, I, var):
	measure_list = np.arange(I + 1)
	acc_list = []
	for d in measure_list:
		acc = (hamming_dist.sum(axis=1) <= d).float().mean().item()
		acc_list.append(var + [d, acc, hamming_dist.mean().item()])
	return acc_list


total_btsp, total_btsp2, total_mfn, total_cfn, total_cfn2, total_mean = [], [], [], [], [], []
q_num = int(n * fq)
I = int(m * fp)


def avg_pattern_distance(z, M, precison=torch.float16):
	if z.shape[0] != M:
		z = z.T
	assert z.shape[0] == M
	z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
	avg_distance = (z_dist.float().sum()) / M / (M - 1) / z.shape[1]
	return avg_distance.item()


def avg_assembly_overlap(z, M, precison=torch.float32):
	if z.shape[0] != M:
		z = z.T
	assert z.shape[0] == M
	z_dist = z.to(precison) @ z.to(precison).T
	avg_distance = (z_dist.float().sum() - z_dist.diag().sum()) / M / (M - 1)
	return avg_distance.item()


def important_parameter(z1, z2, z3, M):
	z1, z2, z3 = z1.float(), z2.float(), z3.float()
	assemble_size = z1.sum(1).mean()
	z_mask_overlap = ((z1 * z2).sum(0) / (z1.sum(0) + 1)).mean()
	z_pattern_overlap = avg_assembly_overlap(z1, M)
	return [assemble_size.item(), z2.sum(1).mean().item(), z3.sum(1).mean().item(), z_mask_overlap.item(),
	        z_pattern_overlap]

 
vth1 = vth2 = vth_b = int(I * p_w * 0.45)
w_range = 8
weight_matrix = torch.randint(-w_range, w_range + 1, (1, n)).cuda().float()
vth_threshold = 0
linear_classifier = lambda x: (weight_matrix @ x.float().T > vth_threshold)

for idx_var1, var1 in enumerate(variable_list1):
	M = int(var1)
	precison = torch.float16
	start_time = time.time()
	# generate datasets
	arrays = np.arange(0, m - 1)
	data = torch.Tensor(np.random.binomial(n=1, p=p, size=(m, M))).to(device).to(precison)
	val_data = torch.Tensor(np.random.binomial(n=1, p=p, size=(m, 100))).to(device).to(precison)
	"""
    training process
    """
	# divide ca1 into two non-overlap subsets
	W = 0
	fq_half = fq / 2
	fq_ltp = fq_half
	fq_ltd = fq_half
	W_mask = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
	plateau_winner1 = (torch.rand(M, 1, n).to(device) <= fq_ltp).to(precison)
	plateau_winner2 = (torch.rand(M, 1, n).to(device) <= fq_ltd / (1 - fq_ltp)).to(precison)
	plateau_winner2 = plateau_winner2 * (1 - plateau_winner1)  # mutual exclusive
	start_time = time.time()
	
	# note that the occurance of p1 and p2 is mutual exclusive
	for idx in range(M):
		inpx = data[:, idx].reshape(-1, 1).to(precison)
		W = ((W + inpx @ plateau_winner1[idx, :, :]) >= 1) & (
				(W + inpx @ plateau_winner2[idx, :, :]) <= 1)
	W = W_mask * W
	del W_mask
	torch.cuda.empty_cache()
	print('finish the generation!', time.time() - start_time)
	
	for idx_var2, var2 in enumerate(variable_list2):
		fd = var2
		mask_num = int(I * var2)
		start_time = time.time()
		print('\n threshold', vth1, vth2)
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
            calculate the recall rate
            """
			query_data = data.clone().to(device).T
			start_time = time.time()
			k = 0
			
			for i in range(M):
				tmp = query_data[i]

				tmp_idx1 = torch.where(tmp == 1)[0]
				
				mask_num = int(len(tmp_idx1) * fd)
				random_idx1 = np.random.choice(len(tmp_idx1), size=mask_num, replace=False)
				query_data[i, tmp_idx1[random_idx1]] = 0
			
			# random_idx0 = np.random.choice(len(tmp_idx0), size=mask_num, replace=False)
			# query_data[i, tmp_idx0[random_idx0]] = 1
			query_data = query_data.T
			print(time.time() - start_time)
			
			torch.cuda.empty_cache()
			z_raw1 = (W.to(precison).T @ data.to(precison) > vth_b).t()
			z_raw_rate1 = z_raw1.float().mean().item()
			z_query1 = (W.to(precison).T @ query_data.to(precison) > vth_b).t()
			z_val1 = (W.to(precison).T @ val_data.to(precison) > vth_b).t()
			z_query_rate1 = z_query1.float().mean().item()
			z = z_raw1
			z_dist = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
			(z_dist.float().mean()) / M / (M - 1)
			
			vth_r = vth2
			
			z_raw2 = (W.to(precison).T @ data.to(precison) > vth_r).t()
			z_query2 = (W.to(precison).T @ query_data.to(precison) > vth_r).t()
			hamming_dist2 = abs(z_raw2.float() - z_query2.float())
			rp_mean2 = hamming_dist2.float().mean().item()
			
			z_val2 = (W.to(precison).T @ val_data.to(precison) > vth_r).t()
			
			import_param1 = important_parameter(z_raw1, z_query1, z_val1, M)
			import_param2 = important_parameter(z_raw2, z_query2, z_val2, M)
			
			"""
            predict
            """
			if trial == 0:
				estimate_hamming1 = 1
				estimate_hamming2 = 1
				var = [vth1] + [vth2] + [estimate_hamming1] + [estimate_hamming2]
				fr1_pattern = 1
				fr2_pattern = 1
			
			btsp_mean1 = abs(z_raw1.float() - z_query1.float()).float().mean().item()
			btsp_mean2 = abs(z_raw2.float() - z_query2.float()).float().mean().item()
			
			z = z_raw1[::20]  # reduce the computational cost
			z_dist1 = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
			zab1 = (z_dist1.float().mean()).item() / n
			
			z = z_raw2[::20]  # reduce the computational cost
			z_dist2 = (1 - z.to(precison)) @ z.to(precison).T + z.to(precison) @ (1 - z.to(precison)).T
			zab2 = (z_dist2.float().mean()).item() / n
			
			torch.cuda.empty_cache()
			
			class_raw1 = linear_classifier(z_raw1)
			class_query1 = linear_classifier(z_query1)
			class_raw2 = linear_classifier(z_raw2)
			class_query2 = linear_classifier(z_query2)
			print('firing rate', z_raw1.float().mean(), z_query1.float().mean(), z_raw2.float().mean(),
			      z_query2.float().mean())
			
			torch.cuda.empty_cache()
			
			acc1 = (class_raw1 == class_query1)[0].float().mean().float().item()
			acc2 = (class_raw2 == class_query2)[0].float().mean().float().item()
			zaa1 = btsp_mean1
			zaa2 = btsp_mean2
			
			start_time = time.time()
			
			"""
			Record the results
			"""
			# 6-8 import param1;
			# 9-11: important param2
			# 6:
			pattern_dist = import_param1 + import_param2
			
			ratios1 = zaa1 / (zab1 + 1e-20)
			ratios2 = zaa2 / (zab2 + 1e-20)
			
			btsp_err1 = abs(z_raw1.float() - z_query1.float())
			btsp_err2 = abs(z_raw2.float() - z_query2.float())
			f_ratio = lambda x, thr: (x.mean(1) <= thr).float().sum().item()
			r1 = [f_ratio(btsp_err1 / (zab1 + 1e-20), 0.1), f_ratio(btsp_err2 / (zab2 + 1e-20), 0.1),
			      f_ratio(btsp_err1 / (zab1 + 1e-20), 0.25), f_ratio(btsp_err2 / (zab2 + 1e-20), 0.25)]
			
			del z_dist1, z_dist2, z_raw1, z_raw2
			record.append([var1, var2,
			               vth1, vth2,
			               btsp_mean1, btsp_mean2, ] + pattern_dist + r1 + [zaa1, zaa2, zab1, zab2,
			                                                                ratios1, ratios2, acc1, acc2])
			
			print('Stored patterns', M, 'masking', var2, 'Comparison of \n P1: assembly size(3) + overlap',
			      import_param1, ' \n P2: assembly size(3) + overlap', import_param2)
			print('ratio:', ratios1, ratios2)
		# record experimental results
		print('\nrecord', record)
 