import os
node_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = node_id
import numpy as np
import torch
import numpy
import matplotlib
import copy

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

device = torch.device('cuda')
from datetime import date

m = int(2.5e4)  # Input size
n = int(3.9e4)  # Output size
fq = 0.005  # Plateau probability
fq_half = fq / 2
fp = 0.005
p_w = 0.6
data_name = 'feedback_masking_' + date.today().strftime("%d-%m") + '_pw_' + str(p_w) + 'fq' + str(
	fq) + '_hebbfb_fixedthr_independentM_n' + str(n) + node_id + '.mat'


def fix_random_seeds(seed_value=0):
	np.random.seed(seed_value)  # Fix random seed for numpy
	torch.random.seed()


fix_random_seeds(1111)


def np2torch(x):
	return torch.from_numpy(x).cuda()


def torch2np(x):
	return x.cpu().detach().numpy()


num_img_list = np.arange(2e3, 3e4, 2e3)
num_mask_list = np.arange(0.0, 0.96, 0.02)

records = []

# obtained by linear fitting
coef1 = [0.0005035858877964149, -3.461460303565554, 19.986813186813187]
coef2 = [0.0007538172353961835, -23.53524142997826, 19.104164256795826]
f_thr1 = lambda x, y: int(coef1[0] * x + coef1[1] * y + coef1[2])
f_thr2 = lambda x, y: int(coef2[0] * x + coef2[1] * y + coef2[2])

import time

precison = torch.float16
record_overall = []
precison = torch.float32
for trial in range(3):
	records = []
	for num_images in num_img_list:
		# Initialize weight matrices
		print('\n\n num_images', num_images)
		num_images = M = int(num_images)
		# note the X is a transpose of (m,M)!
		X = torch.Tensor(np.random.binomial(n=1, p=fp, size=(m, num_images))).cuda().to(precison).T
		plateaus = (torch.rand(num_images, n).cuda() <= fq_half).to(precison)
		sum_W = X.T @ plateaus
		W_feed_init = 0.
		W_back_init = 0.
		W_feed_init_control = 0.
		
		sum_W = X.T @ plateaus
		W_mask1 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
		W_mask2 = (torch.rand(m, n).to(device) <= p_w).bool().to(device)
		image_intensity = X.sum(1).mean()
		thr_ca1 = int(m * fp * p_w * 0.6)
		thr_ca2 = int(m * fp * p_w * 0.6)
		## select one threshold for learning
		# Method1:  fast simulation (recommended)
		W_feed_init = X.T @ plateaus
		W_feed_init = (W_feed_init * W_mask1) % 2
		y_sum = X @ W_feed_init
		spikes1 = (y_sum > thr_ca1).to(precison)
		W_back_init += spikes1.T @ X

		W_feed = W_feed_init * W_mask1
		W_back = W_back_init * W_mask2.T
		W_back = (W_back >= 1).to(precison)
		
		# Method2: slow simulation for running each sample
		# fq_half = fq / 2
		# fq_ltp = fq_half
		# fq_ltd = fq_half
		# plateau_winner1 = (torch.rand(M, 1, n).to(device) <= fq_ltp).to(precison)
		# plateau_winner2 = (torch.rand(M, 1, n).to(device) <= fq_ltd / (1 - fq_ltp)).to(precison)
		# for sample_idx in range(num_images):
		# 	# step 1
		# 	inpx = X[sample_idx].reshape(-1, 1).to(precison)
		# 	W_feed_init = ((W_feed_init + inpx @ plateau_winner1[sample_idx, :, :]) >= 1) & (
		# 			(W_feed_init + inpx @ plateau_winner2[sample_idx, :, :]) <= 1)
		#
		# 	W_feed_init = W_feed_init % 2
		# 	# step 2
		# 	y_sum = W_feed_init.T @ X[sample_idx].reshape(-1, 1)
		# 	spikes1 = (y_sum > thr_ca1).float()
		# 	spikes2 = (y_sum > thr_ca2).float()
		# 	W_back_init += spikes1 @ X[sample_idx].reshape(1, -1)
		
		W_feed = W_feed_init * W_mask1
		W_back = W_back_init * W_mask2.T
		
		del W_mask1, W_mask2, W_back_init, W_feed_init, spikes1, plateaus
		torch.cuda.empty_cache()

		for masked_ratio in num_mask_list:
			# Mask the top half of the patterns and project the results using Wf
			fd = masked_ratio
			start_time = time.time()
			k = 0
			
			X_masked = copy.deepcopy(X).to(device)
			X_masked[:, : int(m * masked_ratio)] = 0
			# print(time.time() - start_time)
			
			zab = 2 * m * (1 - X.mean()) * X.mean()
			zab = zab.item()
			err1 = (X - X_masked).abs().mean()
			raw_err = err1.item()
			"""
			BTSP
			"""
			input_sum_ca1 = X_masked @ W_feed
			# v1
			reconstruct_results = []
			if masked_ratio <= 0.65:
				fixed_M = 2e4
				opt_thr1_fitting = f_thr2(fixed_M, masked_ratio)
				opt_thr3_fitting = f_thr1(fixed_M, masked_ratio)
			
			for thr_ca1 in [opt_thr1_fitting] * 2:
				y_ = (input_sum_ca1 >= thr_ca1).to(precison)
				X_projected = y_ @ W_back
				for thr_ca3 in [opt_thr3_fitting] * 2:
					tmp = (X_projected >= thr_ca3).to(precison)
					err0 = (tmp - X).abs().mean()
					items = [thr_ca1, thr_ca3, err0.item(), err1.item()]
					reconstruct_results.append(items)
			
			reconstruct_array = np.array(reconstruct_results)
			idx_min_err = reconstruct_array[:, 2].argmin()
			opt_thr_ca1, opt_thr_ca3, opt_err = reconstruct_array[idx_min_err][:3]
			print('masking ratio',masked_ratio,'reconstruction error:',opt_err,opt_err / (raw_err + 1e-4))
 