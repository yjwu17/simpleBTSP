import time
from numba import jit, int32
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import kstest, anderson, shapiro
import numpy as np
import math
import scipy.special as sc
# from threshold_tools import *
from statistics import NormalDist as NDist
import random
import torch
from utils import *
from scipy.stats import hypergeom


ERR_TOL_2 = 1e-10
ERR_TOL_1 = 1e-250
ERR_TOL  = 1e-50
BIAS = 0.2
SCALE_STDP = 1.5
CA1_FIRING_RATE = 0.03


def setup_seed(seed):
	# print('set seed',seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

@jit
def NormalDist(mu,sigma):
	return NDist(mu=mu, sigma=sigma)

@jit
def gamma_comb(n, k):
	f = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
	return f

@jit
def prob_on(p, q, N1):
	return 0.5 + 0.5 * (1 - 2 * p) ** N1


def log_prob_on(p, q, N1):
	tmp = N1*math.log((1 - 2 * p))+math.log(0.5)
	return 0.5 + math.exp(tmp)

@jit
def prob_off(p, q, N1):
	return 0.5 - 0.5 * (1 - 2 * p) ** N1

@jit
def binomal(n, k, p):
	return math.comb(n, k) * p ** k * (1 - p) ** (n - k)

@jit
def n_prods(N1, lambdas):
	# compute the fractorials for a large value
	f = math.exp(-lambdas)
	for factor in range(1,N1+1):
		if factor==0:
			factor = 1
		f *= lambdas / factor
	return f

@jit
def gamma_binomal(n, k, p0):
	if p0 > 0:
		if n < 200:
			return binomal(n,k,p0)
		p = max(min(p0,1),0)
		if n-k+1 > 0:
			log_value = math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log(1-p)
			return np.exp(log_value)
		else:
			return 0.
	else:
		if k == 0:
			return 1
		else:
			return 0.


@jit
def combin_prob(n, k):
	f = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
	return f

@jit
def hypergem(m,I,d, alpha):
	f = math.comb(I,alpha) * np.exp(combin_prob(m-I,d-alpha) - combin_prob(m,d))

	# f = math.comb(I,alpha) * np.exp(combin_prob(m-I,d-alpha)) / np.exp( combin_prob(m,d))
	return f

@jit
def gamma_beta_less(I, k, p):
	if k <= I:
		f0 = sc.betainc(I - k, k + 1, 1 - p) * sc.beta(I - k, k + 1)
		if f0 > 0:
			f = gamma_comb(I, k) + np.log(f0)
			return (I - k) * np.exp(f)
		else:
			return 0
	else:
		return 1

@jit
def gamma_greater_less(I, k, p):
	p = max(min(p, 1), 0)
	f0 = sc.betainc(k, I - k + 1, p) * sc.beta(k, I - k + 1)
	if f0>0:
		f = gamma_comb(I, k) + np.log(f0)
		return k * np.exp(f)
	else:
		return 0


def get_I_prob( m,  fp,criterial=ERR_TOL_2, iters = 1 ):
	I_tmp = [ (idx , binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = []
	p_test = 0.
	if iters == 1:
		for i, prob in I_tmp:
			if prob > criterial:
				I_prob.append([i,prob])
				p_test += prob
	elif iters>1:
		for i, prob in I_tmp:
			if prob > criterial and i % iters == 0 :
				I_prob.append([i,prob*iters])
				p_test += prob
	return I_prob

from scipy.stats import binom

def Fb(n,k,p):
	n = int(n)
	if n<0:
		return 0
	else:
		if k <= n:
			return binom.cdf(k,n,p)
		else:
			return 1
 

def hamming_accuracy(hamming_dist, I, var):
    measure_list = np.arange(I+1)
    acc_list = []
    for d in measure_list:
        acc = (hamming_dist.sum(axis=1) <= d).float().mean().item()
        acc_list.append(var + [d, acc])
    return acc_list

def p_g(I, k, M, n, p, q, basic_prop_list = None):
	p0, p1 = 0., 0.
	vth = k
	lambdas = q * M
	M_max = int(lambdas * 4 )
	basic_prop_list = [n_prods(x, lambdas) for x in range(0, M_max + 1)]
	for N1 in range(M_max):
		p_on = prob_on(p, 0, N1)
		prob_N1 = basic_prop_list[N1]

		if p_on == 1:
			p1 += prob_N1
		else:
			p1 += prob_N1 * (1 - Fb(I, vth, p_on))

		if N1 > lambdas * SCALE_STDP and prob_N1 < ERR_TOL:
			break
	p1 = min(p1, 1)
	return min(p1, 1)


def p_l(I, k, M, n, p, q=None, basic_prop_list = None):
	p1 = 0.
	vth = k
	lambdas = q * M
	M_max = int(lambdas * 4)
	basic_prop_list = [n_prods(x, lambdas) for x in range(0, M_max + 1)]
	for N1 in range(M_max):

		p_off = prob_off(p, 0, N1)
		prob_N1 = basic_prop_list[N1]

		if p_off == 1:
			p1 += 1
		else:
			p1 += prob_N1 * Fb(I, vth, p_off)
	p1= min(p1, 1)
	return min(p1, 1)


def p_g_sparse(I, k, M, n, p, q, p_w):

	vth = k
	lambdas = q * M
	M_max = int(lambdas * 4 )
	basic_prop_list = [(x, gamma_binomal(M, x, q)) for x in range(0, M_max )]
	if len(basic_prop_list) > 50: basic_prop_list =  prob_shuffle(basic_prop_list,jump=2,error_tol = 1e-5)

	I_prop_list = [(x, gamma_binomal(I, x, p_w)) for x in range(I+1)]
	if len(I_prop_list) > 50: I_prop_list = prob_shuffle(I_prop_list,jump=2,error_tol = 1e-6)

	p0, p1 = 0., 0.
	for (N1, prob_N1) in basic_prop_list:
		p_on = prob_on(p, 0, N1)
		p0 = 0.
		for N2, prob_I in I_prop_list:
			if p_on == 1:
				p0 += prob_I

			else:
				p0 += prob_I * (1 - Fb(N2, vth, p_on))
		p1 += prob_N1 * p0


	p1 = min(p1, 1)
	return p1

@jit
def p_l_sparse(I, k, M, n, p, q, p_w):
	p1 = 0.
	vth = k
	lambdas = q * M
	M_max = int(lambdas * 4)
 
	basic_prop_list = [(x, gamma_binomal(M, x, q)) for x in range(0, M_max )]
	if len(basic_prop_list) > 50: basic_prop_list =  prob_shuffle(basic_prop_list,jump=3,error_tol = 1e-7)

	I_prop_list = [(x, gamma_binomal(I, x, p_w)) for x in range(I+1)]
	if len(I_prop_list) > 50: I_prop_list = prob_shuffle(I_prop_list,jump=2,error_tol = 1e-7)

	for (N1, prob_N1) in basic_prop_list:
		p_off = prob_off(p, 0, N1)
		p0 = 0.
		for N2, prob_I in I_prop_list:
			if p_off == 1:
				p0 += prob_I
			else:
				p0 += prob_I * Fb(N2, vth, p_off)
		p1 += prob_N1 * p0
	p1= min(p1, 1)
	return min(p1, 1)

def pg_non_orgthogonal(vth, M, n, fp,fq, I,comm_num, p_w=None ):
	p0, p1 = 0., 0.
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2 )
	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, M_max )]
	basic_prop_list = []
	for item in basic_temp_list:
		if item[1] > 1e-15: basic_prop_list.append(item)

	for (N1, prob_N1) in basic_prop_list:
		p_basic = prob_on(fp, fq, N1)
		if N1 > q_num * SCALE_STDP and prob_N1 < ERR_TOL:
			break
		if N1 % 2 == 0:
			p1 += prob_N1 * (1 - Fb(I-comm_num, vth, p_basic))
		else:
			if p_basic == 1:
				p1 += prob_N1
			else:
				p1 += prob_N1 * (1 - Fb(I-comm_num, vth - comm_num, p_basic))
	p1 = min(p1, 1)
	return min(p1, 1)


def pl_non_orgthogonal(vth, M, n, fp,fq,I,comm_num):
	p0, p1 = 0., 0.
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2 )
	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, M_max )]
	basic_prop_list = []
	for item in basic_temp_list:
		if item[1] > 1e-15: basic_prop_list.append(item)
	for (N1, prob_N1) in basic_prop_list:
		p_basic = prob_off(fp, fq, N1)
		if N1 % 2 == 0:
			if p_basic == 1:
				p1 += 1
			else:
				p1 += prob_N1 * Fb(I-comm_num, vth, p_basic)
		else:
			if p_basic == 1:
				p1 += 1
			else:
				# print(prob_N1,Fb(I - comm_num, vth - comm_num, p_basic))
				p1 += prob_N1 * Fb(I-comm_num, vth - comm_num, p_basic)
			# p1 += prob_N1 * Fb(I, vth - comm_num, p_basic)
	p1 = min(p1, 1)
	return min(p1, 1)



def pg_non_orgthogonal_sparse(vth, M, n, fp,fq, I,comm_num, p_w=None ):
	# assume I is the overall number of 1s bits
	p0, p1 = 0., 0.
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2 )

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, M_max )]
	basic_prop_list = prob_shuffle(basic_temp_list,jump=3,error_tol = 1e-8)

	basic_commm_list = [(x, binomal(comm_num, x, p_w)) for x in range(0, comm_num+1)]
	if len(basic_commm_list) > 50:basic_commm_list=prob_shuffle(basic_commm_list, jump=2)

	# basic_temp_list = [(x, binomal(I-comm_num, x, p_w)) for x in range(0, I-comm_num+1)]
	basic_I_list = [(x, binomal(I-comm_num, x, p_w)) for x in range(0, I-comm_num+1)]
	if len(basic_I_list) > 50: basic_I_list = prob_shuffle(basic_I_list, jump=2)

	p_final = 0.
	for (N1, prob_N1) in basic_prop_list:
		p_final_ = 0.
		p_basic = prob_on(fp, fq, N1)
		for (I_com_avg, prob_I0) in basic_I_list:
			p1 = 0.
			for (cnums, prob_I1) in basic_commm_list:
				if N1 > q_num * SCALE_STDP and prob_N1 < ERR_TOL:
					break
				if N1 % 2 == 0:
					p1 += prob_I1 * (1 - Fb(I_com_avg, vth, p_basic))
				else:
					if p_basic == 1:
						p1 += prob_I1
					else:
						p1 += prob_I1 * (1 - Fb(I_com_avg, vth - cnums, p_basic))
			p_final_ += prob_I0 * p1
		p_final += prob_N1 * p_final_
	p_final = min(p_final, 1)
	return p_final

def pl_non_orgthogonal_sparse(vth, M, n, fp,fq,I,comm_num,p_w ):
	p0, p1 = 0., 0.
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2 )

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, M_max )]
	basic_prop_list = prob_shuffle(basic_temp_list,jump=3,error_tol = 1e-8)

	basic_commm_list = [(x, binomal(comm_num, x, p_w)) for x in range(0, comm_num+1)]
	if len(basic_commm_list) > 50:basic_commm_list=prob_shuffle(basic_commm_list, jump=2)

	# basic_temp_list = [(x, binomal(I-comm_num, x, p_w)) for x in range(0, I-comm_num+1)]
	basic_I_list = [(x, binomal(I-comm_num, x, p_w)) for x in range(0, I-comm_num+1)]
	if len(basic_I_list) > 50: basic_I_list = prob_shuffle(basic_I_list, jump=2)

	p_final = 0
	for (N1, prob_N1) in basic_prop_list :
		p2 = 0.
		p_basic = prob_off(fp, fq, N1)
		for (I_com_avg, prob_I0) in basic_I_list:
			p1 = 0.
			for (cnums, prob_I1) in basic_commm_list:
				if N1 > q_num * SCALE_STDP and prob_N1 < ERR_TOL:
					break
				if N1 % 2 == 0:
					p1 += prob_I1 * Fb(I_com_avg, vth, p_basic)
				else:
					if p_basic == 1:
						p1 += prob_I1
					else:
						p1 += prob_I1 * Fb(I_com_avg, vth - cnums, p_basic)
			p2 += prob_I0 * p1
		p_final += prob_N1 * p2
	p_final = min(p_final , 1)
	return p_final



def btsp_firing(I, vth, M, n,m, fp, fq, p_w, criterial = ERR_TOL_2):
	I_tmp = [ (idx, gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(2*I))]
	I_prob = prob_shuffle(I_tmp,jump=3, error_tol = 1e-5 )
	p_final =0
	Perr = 0
	p_on = prob_on(fp * fq, 0, M - 1)
	p_off = prob_off(fp * fq, 0, M - 1)
	for I1, prob_I in I_prob:
		p_f1 = Fb(I1, vth, p_on)
		p_f2 = Fb(I1, vth, p_off)
		p_firing = fq * (1 - p_f1) + (1 - fq) * (1 - p_f2)
		p_final += prob_I * p_firing
	return p_final


def sparse_overlap_simple(I, vth, M, n,m, fp, fq,p_w):

	q_tmp = [ (idx , gamma_binomal(n, n_q, fq  )) for (idx, n_q) in enumerate(range(int(3*n*fq)))]
	I_prob = [(I, 1.)]
	q_prob = prob_shuffle(q_tmp,jump=2,error_tol = 1e-8)
	p_on = prob_on(fp * fq, 0, M - 1)
	p_off = prob_off(fp * fq, 0, M - 1)

	pg1 = 1 - Fb(I, vth, p_w * p_on)
	pl1 = 1 - Fb(I, vth, p_w * p_off)

	p0, p1 = 0., 0.
	p_avg = fq * pg1 + (1 - fq) * pl1
	p0  =   (1 - 2 * p_avg * (1 - p_avg)) ** n
	for j0, prob_q in q_prob:
		n_q = j0
		pa = pg1 * p_avg + (1 - pg1) * (1 - p_avg)
		pb = pl1 * p_avg + (1 - pl1) * (1 - p_avg)
		p1 += prob_q * pa ** n_q * pb ** (n - n_q)

	return p0, p1



def sparse_overlap_prob(I, vth, M, n,m, fp, fq,p_w):
	q = fq
	I_tmp = [ (idx, gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(2*I))]
	q_tmp = [ (idx , gamma_binomal(n, n_q, fq  )) for (idx, n_q) in enumerate(range(int(3*n*fq)))]
	I_prob = prob_shuffle(I_tmp,jump=3, error_tol = 1e-5 )
	q_prob = prob_shuffle(q_tmp,jump=3,error_tol = 1e-5)
	p_on = prob_on(fp * fq, 0, M - 1)
	p_off = prob_off(fp * fq, 0, M - 1)

	pg_list = [(x[0], 1 - Fb(x[0], vth, p_on)) for (I1,x) in enumerate(I_prob)]
	pl_list = [(x[0],  Fb(x[0], vth, p_off)) for (I1, x) in enumerate(I_prob)]

	pg_list_ = [(x[0], p_g_sparse(I=x[0], k=vth, M=M, n=n, p=fp, q=q, p_w=p_w)) for (i,x) in enumerate(I_prob)]
	pl_list_ = [(x[0], p_l_sparse(I=x[0], k=vth, M=M, n=n, p=fp, q=q, p_w=p_w)) for (i, x) in enumerate(I_prob)]
	p_final = 0.
	for j0, prob_q in q_prob:
		n_q = j0
		p0, p1 = 0., 0.
		for j1, item in enumerate(I_prob):
			i1, prob_i1 = item
			pg1 = pg_list[j1][1]
			pl1 = pl_list[j1][1]
			p_o1 = q * pg1 + (1 - q) * (1 - pl1)
			for j2, item in enumerate(I_prob):
				i2, prob_i2 = item
				prob_factor = prob_i1 * prob_i2
				if prob_factor * prob_q > 1e-11:
					pg2 = pg_list[j2][1]
					pl2 = pl_list[j2][1]
					pa = pg2 * p_o1 + (1 - pg2) * (1 - p_o1)
					pb = (1 - pl2) * p_o1 + pl2 * (1 - p_o1)
					p_temp1 = prob_factor * pa ** n_q * pb ** (n - n_q)
					p0 += p_temp1
		p_final +=  prob_q * p0
	return p_final




def sparse_overlap_orthogonal(I, vth, M, n,m, p, fq, p_w, criterial = ERR_TOL_2):
	q = fq
	I_tmp = [ (idx, gamma_binomal(m, n_I, p)) for (idx, n_I) in enumerate(range(2*I))]
	q_tmp = [ (idx , gamma_binomal(n, n_q, q  )) for (idx, n_q) in enumerate(range(int(3*n*fq)))]
	# I_prob = prob_shuffle(I_tmp,jump=3, error_tol = 1e-5 )
	I_prob = [(I,1.)]
	q_prob = prob_shuffle(q_tmp,jump=3,error_tol = 1e-5)
	pg_list = [(x[0], p_g_sparse(I=x[0], k=vth, M=M, n=n, p=p, q=q, p_w=p_w)) for (i,x) in enumerate(I_prob)]
	pl_list = [(x[0], p_l_sparse(I=x[0], k=vth, M=M, n=n, p=p, q=q, p_w=p_w)) for (i, x) in enumerate(I_prob)]
	p_final = 0.
	for j0, prob_q in q_prob:
		n_q = j0
		p0, p1 = 0., 0.
		for j1, item in enumerate(I_prob):
			i1, prob_i1 = item
			pg1 = pg_list[j1][1]
			pl1 = pl_list[j1][1]
			p_o1 = q * pg1 + (1 - q) * (1 - pl1)
			for j2, item in enumerate(I_prob):
				i2, prob_i2 = item
				prob_factor = prob_i1 * prob_i2
				if prob_factor * prob_q > 1e-11:
					pg2 = pg_list[j2][1]
					pl2 = pl_list[j2][1]
					pa = pg2 * p_o1 + (1 - pg2) * (1 - p_o1)
					pb = (1 - pl2) * p_o1 + pl2 * (1 - p_o1)
					p_temp1 = prob_factor * pa ** n_q * pb ** (n - n_q)
					p0 += p_temp1
		p_final +=  prob_q * p0
	return p_final




def overlap_nonorthogonal_sparse_jumps(I, vth, M, n, m, p, fq, comm_num, p_w ,jumps=3):
	q = fq
	I_tmp = [(idx + comm_num, gamma_binomal(m - comm_num, idx, p)) for (idx, n_I) in enumerate(range(comm_num, 4 * I))]
	q_tmp = [(idx, gamma_binomal(n, n_q, q)) for (idx, n_q) in enumerate(range(int(3 * n * fq)))]
	I_prob = prob_shuffle(raw_list=I_tmp,jump=jumps,error_tol = 1e-8)
	q_prob = prob_shuffle(raw_list=q_tmp,jump=jumps,error_tol = 1e-8)
	pg_list = [(i, pg_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], comm_num,p_w)) for (i, x) in enumerate(I_prob)]
	pl_list = [(i, pl_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], comm_num,p_w)) for (i, x) in enumerate(I_prob)]
	p_final = 0.
	for j0, prob_q in q_prob:
		n_q = j0
		p0, p1 = 0., 0.
		for j1, item in enumerate(I_prob):
			i1, prob_I1 = item
			pg1 = pg_list[j1][1]
			pl1 = pl_list[j1][1]
			p_o1 = q * pg1 + (1 - q) * (1 - pl1)
			for j2, item in enumerate(I_prob):
				i2, prob_I2 = item
				prob_factor = prob_I1 * prob_I2
				if prob_factor * prob_q > 1e-13:
					pg2 = pg_list[j2][1]
					pl2 = pl_list[j2][1]
					pa = pg2 * p_o1 + (1 - pg2) * (1 - p_o1)
					pb = (1 - pl2) * p_o1 + pl2 * (1 - p_o1)
					p_temp1 = prob_factor * pa ** n_q * pb ** (n - n_q)
					p0 += p_temp1
		p_final += prob_q * p0
	return p_final

def overlap_nonorthogonal_sparse(I, vth, M, n, m, p, fq, comm_num, p_w=1.,iters=1):
	q = fq
	I_tmp = [(idx + comm_num, gamma_binomal(m - comm_num, idx, p)) for (idx, n_I) in enumerate(range(comm_num, 4 * I))]
	q_tmp = [(idx, gamma_binomal(n, n_q, q)) for (idx, n_q) in enumerate(range(int(3 * n * fq)))]
	criterial = ERR_TOL_2
	I_prob, I_sum = [], []
	for i, prob in I_tmp:
		if prob > criterial :
			I_prob.append([i, prob ])
	q_prob = []
	sum_p = 0.
	for i, prob in q_tmp:
		if prob > criterial and i % iters == 0:
			q_prob.append([i, prob*iters])
			sum_p += prob*iters
	if abs(1-sum_p)>1e-4:
		print('Error: too large jumps')

	pg_list = [(i, pg_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], comm_num,p_w)) for (i, x) in enumerate(I_prob)]
	pl_list = [(i, pl_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], comm_num,p_w)) for (i, x) in enumerate(I_prob)]
	p_final = 0.
	for j0, prob_q in q_prob:
		n_q = j0
		p0, p1 = 0., 0.
		for j1, item in enumerate(I_prob):
			i1, prob_I1 = item
			pg1 = pg_list[j1][1]
			pl1 = pl_list[j1][1]
			p_o1 = q * pg1 + (1 - q) * (1 - pl1)
			for j2, item in enumerate(I_prob):
				i2, prob_I2 = item
				prob_factor = prob_I1 * prob_I2
				if prob_factor * prob_q > 1e-13:
					pg2 = pg_list[j2][1]
					pl2 = pl_list[j2][1]
					pa = pg2 * p_o1 + (1 - pg2) * (1 - p_o1)
					pb = (1 - pl2) * p_o1 + pl2 * (1 - p_o1)
					p_temp1 = prob_factor * pa ** n_q * pb ** (n - n_q)
					p_temp2 = np.log(prob_factor) + (n - n_q) * np.log(pb) + n_q * np.log(pa)
					p0 += p_temp1
					p1 += np.exp(p_temp2)
		p_final += prob_q * p0
	return p_final

def mask_prob_hamming(I, vth, M, m, n, fp, fq, f_d, p_w,err_tol=1e-5):
	I_prob = get_I_prob(m, fp,1e-4)
	d = int(m * f_d)
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2)
	basic_prop_list = [n_prods(x, lambdas) for x in range(M_max + 1)]

	p_final = 0.
	for I1, prob_i1 in I_prob:
		# print(I1)
		P0 = 0.
		alpha = d
		if I1 > d:
			# vmax = min(I1 - alpha + 1, vth)
			vmax = I1 - alpha + 1
			P1 = 0.
			for N1 in range(M_max):
				p_on = prob_on(fp, fq, N1)
				p_off = prob_off(fp, fq, N1)
				prob_N1 = basic_prop_list[N1]
				if N1 > q_num * SCALE_STDP and prob_N1 < err_tol:
					break
				p0, p1 = 0., 0.,
				default_value = 1.
				for v in range(vmax):
					f_on = binomal(I1 - alpha, v, p_on)
					f_off = binomal(I1 - alpha, v, p_off)
					if alpha >= vth - v:
						F_alpha1 = Fb(alpha, vth - v, p_on)
						F_alpha2 = Fb(alpha, vth - v, p_off)
					else:
						F_alpha1, F_alpha2 = 1., 1.,

					# if the selected node belongs to Q(x), do
					tmp0 = f_on * (1 - F_alpha1)
					# if the selected node does not belongs to Q(x), do
					tmp1 = f_off * (1 - F_alpha2)
					p0 += tmp0
					p1 += tmp1

				P1 += prob_N1 * (fq * p0 + (1 - fq) * p1)
			p_final += prob_i1 * P1
	return p_final

def mask_prob_hamming_v3(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	error_tols = 1e-8
	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1) # supp(x)
	mask_num = mask_num

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	basic_prop_list = prob_shuffle(raw_list=basic_temp_list, jump=5, error_tol=error_tols)
	p_final = 0.
	for I1, prob_i1 in I_prob:
		if I1 > mask_num:
			basic_I_pw_list = [(x, binomal(I1, x, p_w)) for x in range(I1 + 1)]
			if len(basic_I_pw_list) > 50:
				basic_I_pw_list = prob_shuffle(basic_I_pw_list, jump=2)
			p0 = 0.
			for I2, prob_i2 in basic_I_pw_list:
				p1 = 0.
				for N1,prob_N1 in basic_prop_list:
					p_on = prob_on(fp, None, N1)
					p_off = prob_off(fp, None, N1)
					mask_list = [ (idx , gamma_binomal(mask_num, n_w, p_w)) for (idx, n_w) in enumerate(range(mask_num+1))]
					p2, p3 = 0., 0.
					for mask_rest, p_rest in mask_list:
						f_on = binomal(I1 - mask_num, vth, p_w * p_on)
						f_off = binomal(I1 - mask_num, vth, p_w * p_off)
						F_alpha1 = Fb( mask_rest, vth , p_on)
						F_alpha2 = Fb( mask_rest, vth , p_off)
						tmp0 = f_on * (1 - F_alpha1)
						tmp1 = f_off * (1 - F_alpha2)
						p2 += p_rest * tmp0
						p3 += p_rest * tmp1
					p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
				p0 += prob_i2 * p1
			p_final += prob_i1 * p0
	return p_final

def mask_prob_hamming_v2(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	error_tols = 1e-8
	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1) # supp(x)
	mask_num = mask_num

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	basic_prop_list = prob_shuffle(raw_list=basic_temp_list, jump=5, error_tol=error_tols)
	p_final = 0.
	for I1, prob_i1 in I_prob:
		if I1 > mask_num:
			basic_I_pw_list = [(x, binomal(I1, x, p_w)) for x in range(I1 + 1)]
			if len(basic_I_pw_list) > 50:
				basic_I_pw_list = prob_shuffle(basic_I_pw_list, jump=2)
			p0 = 0.
			for I2, prob_i2 in basic_I_pw_list:
				p1 = 0.
				for N1,prob_N1 in basic_prop_list:
					p_on = prob_on(fp, None, N1)
					p_off = prob_off(fp, None, N1)
					mask_list = [ (idx , gamma_binomal(mask_num, n_w, p_w)) for (idx, n_w) in enumerate(range(mask_num+1))]
					p2, p3 = 0., 0.
					for mask_rest, p_rest in mask_list:
						f_on = binomal(I1 - mask_num, vth, p_on)
						f_off = binomal(I1 - mask_num, vth, p_off)
						F_alpha1 = Fb(I1 - mask_num + mask_rest, vth , p_on)
						F_alpha2 = Fb(I1 - mask_num + mask_rest, vth , p_off)
						tmp0 = f_on * (1 - F_alpha1)
						tmp1 = f_off * (1 - F_alpha2)
						p2 += p_rest * tmp0
						p3 += p_rest * tmp1
					p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
				p0 += prob_i2 * p1
			p_final += prob_i1 * p0
	return p_final


def sparse_prob_hamming(I, vth, M, m, n, fp, fq, mask_num, p_w, jumps=7):
	error_tols = 1e-8
	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1)

	mask_num = mask_num

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))]
	basic_prop_list = prob_shuffle(raw_list=basic_temp_list, jump=5, error_tol=error_tols)
	p_final = 0.
	for I1, prob_i1 in I_prob:
		basic_I_pw_list = [(x, binomal(I1, x, p_w)) for x in range(I1 + 1)]
		if len(basic_I_pw_list) > 50:
			basic_I_pw_list = prob_shuffle(basic_I_pw_list, jump=2)
		p0 = 0.
		for I2, prob_i2 in basic_I_pw_list:
			p1 = 0.
			for N1,prob_N1 in basic_prop_list:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				vmax = min(I2 - mask_num + 1, vth+1)
				mask_list = [ (idx , gamma_binomal(mask_num, n_w, p_w)) for (idx, n_w) in enumerate(range(mask_num+1))]
				p2, p3 = 0., 0.
				for mask_rest, p_rest in mask_list:
					p_case1, p_case2 = 0., 0.,
					for v in range(0,vmax):
						f_on = binomal(I1 - mask_num, v, p_on)
						f_off = binomal(I1 - mask_num, v, p_off)
						if mask_rest >= vth - v:
							F_alpha1 = Fb(mask_rest, vth - v, p_on)
							F_alpha2 = Fb(mask_rest, vth - v, p_off)
						else:
							F_alpha1, F_alpha2 = 1,1.
						tmp0 = f_on * (1 - F_alpha1)
						tmp1 = f_off * (1 - F_alpha2)
						p_case1 += tmp0
						p_case2 += tmp1
					p2 += p_rest * p_case1
					p3 += p_rest * p_case2
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)

			p0 += prob_i2 * p1
	p_final += prob_i1 * p0
	return p_final

def prob_hamming(I, vth, M, m, n, fp, fq, f_d,err_tol=1e-5):
	1
	I_prob = get_I_prob(m, fp,1e-4)
	d = int(m * f_d)
	lambdas = fq * M
	q_num = n * fq
	M_max = int(lambdas * 2)
	basic_prop_list = [n_prods(x, lambdas) for x in range(M_max + 1)]

	p_final = 0.
	for I1, prob_i1 in I_prob:
		# print(I1)
		P0 = 0.
		for alpha in range(d):
			beta = d - alpha
			prob_alpha = binomal(I1, alpha, f_d)
			# prob_alpha = hypergem(m,I,d,alpha)
			if prob_alpha * prob_i1 > 1e-5:
				vmax = I1 - alpha + 1
				P1 = 0.
				for N1 in range(M_max):
					p_on = prob_on(fp, fq, N1)
					p_off = prob_off(fp, fq, N1)
					prob_N1 = basic_prop_list[N1]
					if N1 > q_num * SCALE_STDP and prob_N1 * prob_alpha < err_tol:
						break
					p0, p1 = 0., 0.,
					default_value = 1.
					for v in range(vmax):
						f_on = binomal(I1 - alpha, v, p_on)
						f_off = binomal(I1 - alpha, v, p_off)
						if alpha >= vth - v:
							F_alpha1 = Fb(alpha, vth - v, p_on)
							F_alpha2 = Fb(alpha, vth - v, p_off)
						else:
							F_alpha1, F_alpha2 = 1,1.

						if beta >= vth - v:
							F_beta1 = Fb(beta, vth - v, p_off)
							F_beta2 = Fb(beta, vth - v, p_off)
						else:
							F_beta1, F_beta2 = 1, 1,
						# if the selected node belongs to Q(x), do
						tmp0 = f_on * (F_alpha1 * (1 - F_beta1) + (1 - F_alpha1) * F_beta1)
						# if the selected node does not belongs to Q(x), do
						tmp1 = f_off * (F_alpha2 * (1 - F_beta2) + (1 - F_alpha2) * F_beta2)
						p0 += tmp0
						p1 += tmp1
					P1 += prob_N1 * (fq * p0 + (1 - fq) * p1)
				P0 += prob_alpha * P1
		p_final += prob_i1 * P0

	return p_final


def rp_firing_rate(vth, m, fp, p_w):
	basic_temp_list = [(x, gamma_binomal(m, x, p_w)) for x in range(1, int(m*p_w*2))]
	synapse_prop_list = prob_shuffle(basic_temp_list, jump=2, error_tol=1e-7)
	p0 = 0.
	for s1, prob_s in synapse_prop_list:
		prob_f = 1 - gamma_beta_less(s1,vth,fp)
		p0 += prob_s * prob_f
	return p0


def grid_search_rp_vth(I, vth_btsp, p_rate_btsp, m, fp,  pw_btsp, p_w = 0.6, jumps=1):
	start_time = time.time()
	x_range = np.arange(int(I * pw_btsp * 0.05), int(I * 0.7), jumps)

	records = []
	for vth in x_range:
		rp_rate = rp_firing_rate(vth, m, fp, pw_btsp*p_w)
		records.append([vth, abs(rp_rate - p_rate_btsp), rp_rate])

	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	optimal_rp_vth = records[arg_idx, 0]
	print('searching results', 'btsp fr', p_rate_btsp, 'random fr', records[arg_idx], optimal_rp_vth, vth_btsp,
		  'time_elapse', time.time() - start_time)
	return optimal_rp_vth

def rp_collision(vth,m, n, fp, p_w):
	basic_temp_list = [(x, gamma_binomal(m, x, p_w)) for x in range(1, int(m*p_w*2))]
	synapse_prop_list = prob_shuffle(basic_temp_list, jump=2, error_tol=1e-7)
	p0 = 0.
	for s1, prob_s in synapse_prop_list:
		prob_f = 1 - gamma_beta_less(s1,vth,fp)
		p0 += prob_s * (prob_f*prob_f)**n
	return p0

 

def rp_nonorthogal_firing_rate(vth,m, fp, p_w, comm_num):
	basic_temp_list = [(x, gamma_binomal(m-comm_num, x, p_w * fp)) for x in range(int(m*p_w*2))]
	synapse_prop_list = prob_shuffle(basic_temp_list, jump=2, error_tol=1e-9)
	common_prop_list = [(x, gamma_binomal(comm_num, x, p_w)) for x in range(int(comm_num)+1)]
	p1 = 0.
	for c1, prob_c in common_prop_list:
		p2 = 1 - gamma_beta_less(m - comm_num, vth - c1, p_w * fp)
		p1 += prob_c * p2
 
	p_final = p1
 
	return p_final
 

def rp_nonorthogal_firing_rate_simple(vth,m, fp, p_w, comm_num,comm_avg):
	basic_temp_list = [(x, gamma_binomal(m-comm_num, x, p_w)) for x in range(int(m*p_w*2))]
	synapse_prop_list = prob_shuffle(basic_temp_list, jump=2, error_tol=1e-7)
	# common_prop_list = [(x, gamma_binomal(comm_num, x, p_w)) for x in range(int(comm_num)+1)]
	# p1 = 0.
	p0 = 0.
	for s1, prob_s in synapse_prop_list:
		prob_f = 1 - gamma_beta_less(s1,vth-comm_avg,fp)
		p0 += prob_s * prob_f
	p1 = p0
	return p1


def grid_search_rp_pw(vth_btsp, p_rate_btsp, m, fp, pw,  comm_num):
	start_time = time.time()
	p_w_list = np.arange(0.,pw,0.00005)
	records = []
	for p_w in p_w_list:
		rp_rate=rp_nonorthogal_firing_rate(vth_btsp, m, fp, p_w, comm_num)
		# if rp_rate >= p_rate_btsp:
		records.append([p_w, abs(rp_rate-p_rate_btsp),rp_rate])
		# if abs(rp_rate-p_rate_btsp) < min(1e-7,p_rate_btsp/2):
		# 	break

	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	print('searching results','btsp p_w', pw, 'random projection p_w',records[arg_idx], 'time_elapse',time.time()-start_time)
	return records[arg_idx, 0]


def grid_search_rp_pw_simple(vth_btsp, p_rate_btsp, m, fp, pw,  comm_num):
	start_time = time.time()
	p_w_list = np.arange(0.,pw, 2e-5)
	comm_avg = int(comm_num * pw)
	records = []
	for p_rp in p_w_list:
		rp_rate=rp_nonorthogal_firing_rate_simple(vth_btsp,m, fp, p_rp, comm_num,comm_avg)
		records.append([p_rp, abs(rp_rate-p_rate_btsp),rp_rate])

	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	print('searching results','btsp p_w', pw, 'random projection p_w',records[arg_idx], 'time_elapse',time.time()-start_time)
	return records[arg_idx, 0]


def grid_search_rp_empirical(vth_btsp, p_rate_btsp, m, n, pw,  data, device,precision):
	p_w_list = np.arange(0.0005,pw,0.001)
	records = []
	for pw_rp in p_w_list:
		w_rp = (torch.rand(m, n).to(device) <= pw_rp).to(precision)
		z_rp = (w_rp.T @ data.to(precision) > vth_btsp).t()
		rp_rate = z_rp.float().mean().item()
		records.append([pw_rp, abs(rp_rate-p_rate_btsp),rp_rate])
		if abs(rp_rate-p_rate_btsp) < min(1e-5,p_rate_btsp/2):
			break

	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	print('searching results','btsp p_w', pw, 'random projection p_w',records[arg_idx])
	return records[arg_idx, 0]


def rp_nonorthogal_collision(vth,m, n, fp, p_w, comm_num):
	basic_temp_list = [(x, gamma_binomal(m-comm_num, x, p_w)) for x in range(int(m*p_w*2))]
	synapse_prop_list = prob_shuffle(basic_temp_list, jump=2, error_tol=1e-7)

	# basic_temp_list = [(x, gamma_binomal(comm_num, x, p_w)) for x in range(int(comm_num)+1)]
	common_prop_list = [(x, gamma_binomal(comm_num, x, p_w)) for x in range(int(comm_num)+1)]
	p1 = 0.
	for comm_ele, prob_comm in common_prop_list:
		p0 = 0.
		for s1, prob_s in synapse_prop_list:
			prob_f = gamma_beta_less(s1,vth-comm_ele,fp)
			p0 += prob_s * (1 - 2*prob_f * (1-prob_f)) ** n
		p1 += prob_comm * p0
		# p1_ += prob_comm * np.exp(p0_)
	# print(p1-p1_,p1,p1_)
	return p1

"""
threshold set
"""

def grid_search_v_min_error(I, m, M, n, fp, fq, p_w=1,step=1,max_firing = 1.,min_firing=None):
	if min_firing is None:
		min_firing = fq*0.1
	x_range = np.arange(int(I * p_w * 0.1), int(I * p_w * 0.9), step)
	if len(x_range) > 200:
		# x_range = np.arange(int(I * p_w * 0.2), min(int(I * p_w * 2), int(I * 0.95)), len(x_range) // 200 + 2)
		x_range = np.arange(int(I * p_w * 0.2), min(int(I * p_w * 0.8), int(I * 0.95)), 1)
	records = []
	for vth in x_range:
		p_on = prob_on(fp * fq, 0, M)
		p_off = prob_off(fp * fq, 0, M)
		p_f1 = Fb(I, vth, p_w * p_on)
		p_f2 = Fb(I, vth, p_w * p_off)
		p_err = fq * p_f1 + (1 - fq) * (1 - p_f2)
		p_firing = fq * (1 - p_f1) + (1- fq) * (1 - p_f2)
		if p_firing < max_firing and p_firing>min_firing:
			records.append([vth, p_err,p_firing])
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	return int(records[:, 0][arg_idx])

@jit
def gridsearch_v_hamming_avg(I, m, M, n, p, q_num, mask_num, p_w, pmax=0.5,pmin = 0.8):
	x_range = np.arange(int(I*p_w*0.2), min(int(I*p_w*2), int(I*0.95)),1)
	if len(x_range) > 1000:
		x_range = np.arange(int(I*p_w*0.2), min(int(I*p_w*2), int(I*0.95)),2)
	records = []
	fq = q_num/n
	fp = p
	for vth in x_range:
		p_on = prob_on(fp * fq, 0, M)
		p_off = prob_off(fp * fq, 0, M)
		p_f1 = Fb(I, vth, p_w * p_on)
		p_f2 = Fb(I, vth, p_w * p_off)
		p_firing = fq * (1 - p_f1) + (1 - fq) * (1 - p_f2)
		if p_firing > fq*pmin and p_firing < pmax:
			p_dist = mask_sparse_hamming_simpler(I, vth, M, m, n, fp, fq, mask_num, p_w )
			print([vth, p_dist, p_firing])
			records.append([vth, p_dist,p_firing])
	if records == []:
		print('pay_attention to the selection!')
		return int(I*p_w*0.5)
	else:
		records = np.array(records)
		arg_idx = np.argmin(records[:, 1])
		print('results:',records )
		return int(records[:, 0][arg_idx])


def gridsearch_v_hamming_ratios(I, m, M, n, p, fq, fd, p_w, pmax=0.15,pmin = 0.05,jumps=1):
	assert fd <= 1  # address some historical prob.
	x_range = np.arange(int(I*p_w*0.1)+1, min(int(I*p_w*2), int(I*0.95)),jumps)
	if len(x_range) > 1000:
		x_range = np.arange(int(I*p_w*0.2), min(int(I*p_w*2), int(I*0.95)),jumps)
	records = []
	fp = p
	k1 = 0
	k2 = 0
	for vth in x_range:
		p_firing = mask_sparse_hamming_firing(I, vth, M, m, n, p, fq, 0, p_w,jumps=2*jumps+1)
		if p_firing > fq*pmin and p_firing < pmax:
			p_dist = mask_sparse_hamming_simple(I, vth, M, m, n, fp, fq, fd, p_w, jumps=2 * jumps - 1)
			p_ratio = p_dist / (1 - p_firing + 1e-50) / (p_firing + 1e-50) / 2
			print([vth, p_ratio, p_firing])
			records.append([vth, p_ratio, p_firing])
			k1 = 0
			k2 = 1
		else:
			k1 = k1 + 1
		if k1 > 5 and k2:
			break;

	if records == []:
		print('pay_attention to the selection!')
		return int(I*p_w*0.5)
	else:
		records = np.array(records)
		arg_idx = np.argmin(records[:, 1])
		print('results:',records )
		return int(records[:, 0][arg_idx])


def gridsearch_v_hamming_ratio_jump(I, m, M, n, p, fq, mask_num, p_w, pmax=0.15,pmin = 0.05,jumps=1):
	jumps = max(int(jumps),2)
	x_range = np.arange(int(I*p_w*0.1), min(int(I*p_w*2), int(I*0.95)), min(jumps,9))

	if len(x_range) > 1000:
		x_range = np.arange(int(I*p_w*0.2), min(int(I*p_w*2), int(I*0.95)), min(jumps,9) )
	records = []
	fp = p
	k1 = 0
	k2 = 0
	for vth in x_range:
		p_firing = mask_sparse_hamming_firing(I, vth, M, m, n, p, fq, 0, p_w,jumps=2*jumps-1)
		if p_firing > fq*pmin and p_firing < pmax:
			p_dist = mask_sparse_hamming_simple_jump(I, vth, M, m, n, fp, fq, mask_num, p_w, jumps=2 * jumps-1)
			p_ratio = p_dist / (1 - p_firing + 1e-50) / (p_firing + 1e-50) / 2
			print([vth, p_ratio, p_firing])
			records.append([vth, p_ratio, p_firing])
			k1 = 0
			k2 = 1
		else:
			k1 = k1 + 1

		if k1 > 3 and k2:
			break;

	if records == []:
		print('pay_attention to the selection!')
		return int(I*p_w*0.5)
	else:
		records = np.array(records)
		arg_idx = np.argmin(records[:, 1])
		print('results:',records )
		return int(records[:, 0][arg_idx])


def gridsearch_v_hamming_ratio_quick(I, m, M, n, p, fq, mask_num, p_w, pmax=0.15,pmin = 0.05,jumps=1):
	jumps = min(jumps,15)
	x_range = np.arange(int(I*p_w*0.3), min(int(I*p_w*1.2), int(I*0.95)),jumps)
	if len(x_range) > 1000:
		x_range = np.arange(int(I*p_w*0.3), min(int(I*p_w*1.2), int(I*0.95)),2*jumps+1 )
	records = []
	fp = p
	k1 = 0
	k2 = 0
	for vth in x_range:
		p_firing = mask_sparse_hamming_firing_v2(I, vth, M, m, n, p, fq, 0, p_w,jumps=2*jumps-1)
		if p_firing > fq*pmin and p_firing < pmax:
			p_dist = mask_sparse_hamming_simple_jump(I, vth, M, m, n, fp, fq, mask_num, p_w, jumps=2 * jumps)
			p_ratio = p_dist / (1 - p_firing + 1e-50) / (p_firing + 1e-50) / 2
			print([vth, p_ratio, p_firing])
			records.append([vth, p_ratio, p_firing])
			k1 = 0
			k2 = 1
		else:
			k1 = k1 + 1

		if k1 > 3 and k2:
			break;

	if records == []:
		print('pay_attention to the selection!')
		return int(I*p_w*0.5)
	else:
		records = np.array(records)
		arg_idx = np.argmin(records[:, 1])
		print('results:',records )
		return int(records[:, 0][arg_idx])

def grid_search_v_avg_collision(I, m, M, n, p, q_num, p_w=1,CA1_FIRING_RATE = CA1_FIRING_RATE):
	start_time = time.time()
	vth_range = np.arange(int(I*p_w*0.5), min(int(I*p_w*2), int(I*0.8)))
	records,record_ = [],[]
	fq = q_num/n
	lambdas = q_num * M / n
	basic_temp_list = [(x  , gamma_binomal(m, x, p)) for x in range(1, int(m*p*2))]
	if len(vth_range) > 50:
		vth_range = vth_range[::len(vth_range)//50]

	for vth in vth_range:
		Perr, P_firing = 0, 0
		prob_I = 1
		p_greater = p_g_sparse(I, vth, M, n, p, fq, p_w)
		p_less = p_l_sparse(I, vth, M, n, p, fq, p_w)
		p_corr = sparse_overlap_orthogonal(I, vth, M, n, m, p, fq, p_w)
		p_firing = fq * p_greater + (1 - fq) * (1 - p_less)
		Perr += prob_I * p_corr
		P_firing += prob_I * p_firing
		record_.append([vth, Perr, P_firing])
		if P_firing < CA1_FIRING_RATE:
			records.append([vth, Perr, P_firing])
			if len(records) > 1:break

	if len(records) > 0:
		records = np.array(records)
		arg_idx = np.argmin(records[:, 1])
		print('Time for grid searching:',  (time.time() - start_time),records[arg_idx]  )
		return int(records[:, 0][arg_idx])
	else:
		print('Warning of threshold')
		return int(I*p_w*0.6)
 

def grid_search_v_nonorthgonal(I, k, M, n, fp, q_num, comm_num, p_w = 1):
	x_range = np.arange(int(I * 0.1), I+comm_num)
	records = []
	fq = q_num/n
	lambdas = q_num * M / n
	M_max = int(lambdas * 3)
	for idx in x_range:
		p_greater = pg_non_orgthogonal(idx, M, n, fp,fq, I,comm_num)
		p_less = pl_non_orgthogonal(idx, M, n, fp,fq, I,comm_num )
		p_err = fq * (1 - p_greater) + (1 - fq) * (1 - p_less)

		records.append([idx, p_err])
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	return int(records[:, 0][arg_idx])

def grid_search_v_nonorthgonal_sparse(I, vth, M, m, n, p_rest, fq, mask_num, p_w, comm_num, jumps=5):
	start_time = time.time()
	x_range = np.arange(int(I * p_w * 0.5), int(I * p_w * 1.1), 1)
	records = []
	for vth in x_range:
		Perr, P_firing = 0, 0
		Perr = mask_hamming_comm_simple(I, vth, M, m, n, p_rest, fq, mask_num, p_w, comm_num, jumps=5)

		records.append([vth, Perr])
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
 
	print('Time for grid searching:',  (time.time() - start_time), records[arg_idx] )
	return int(records[:, 0][arg_idx])


def grid_search_v_nonorthgonal_collison(I, m, M, n, fp, q_num, comm_num, p_w = 1,CA1_FIRING_RATE = 0.05):
	start_time = time.time()
	vth_range = np.arange(3, min(int(I*p_w*3), int(I*0.8)))
	records = []
	fq = q_num/n
	for vth in vth_range:
		p_corr = overlap_nonorthogonal_sparse_jumps(I, vth, M, n, m, fp, fq, comm_num,p_w,jumps=10)
		p_greater = pg_non_orgthogonal_sparse(vth, M, n, fp, fq, I, comm_num, p_w)
		p_less = pl_non_orgthogonal_sparse(vth, M, n, fp, fq, I, comm_num, p_w)
		p_firing = fq * p_greater + (1 - fq) * (1 - p_less)

		if p_firing < CA1_FIRING_RATE:
			records.append([vth,p_corr,p_firing])
		if p_firing < 1e-10:
			break
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	print('Time for grid searching:',  (time.time() - start_time), 'p_corr', records[arg_idx] )
	return int(records[:, 0][arg_idx])



def grid_search_v_max(I, k, M, n, p, q_num):
	x_range = np.arange(int(I * 0.2), int(I * 0.8), 1)
	records = []
	fq = q_num/n
	lambdas = q_num * M / n
	M_max = int(lambdas * 3)
	basic_prop_list = [n_prods(x, lambdas) for x in range(0, M_max + 1)]
	for idx in x_range:
		p_greater = p_g(I=I, k=idx, M=M, n=n, p=p, q=fq, basic_prop_list = basic_prop_list)
		p_less = p_l(I=I, k=idx, M=M, n=n, p=p, q=fq, basic_prop_list = basic_prop_list)
		p_err = fq * (1 - p_greater) + (1 - fq) * (1 - p_less)
		records.append([int(idx//2*2), p_err])
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	return int(records[:, 0][arg_idx])


def grid_search_v_min_plateau(I, k, M, n, p, q_num, p_w=1):
	x_range = np.arange(int(I * 0.025), int(I * 0.85), 1)
	records = []
	fq = q_num/n
	lambdas = q_num * M / n
	M_max = int(lambdas * 3)
	for idx in x_range:
		p_greater = p_g_sparse(I, idx, M, n, p, fq, p_w)
		p_less = p_l_sparse(I, idx, M, n, p, fq, p_w)
		p_err = fq * (1 - p_greater) + (1 - fq) * (1 - p_less)
		records.append([idx, p_err, ])
	records = np.array(records)
	arg_idx = np.argmin(records[:, 1])
	return int(records[:, 0][arg_idx])



"""
new updates
"""

def pg_nonorgthogonal_sparse_coll(vth, M, n, fp,fq, I, I_, cnums, p_w, basic_I_list, basic_prop_list ):
	# assume I is the overall number of 1s bits
	I_rest_2 = int(I_*p_w)
	p_final = 0.
	for (I_rest, prob_I0) in basic_I_list:
		p1 = 0.
		for (N1, prob_N1) in basic_prop_list:
			pon = prob_on(fp, None, N1)
			poff = prob_off(fp, None, N1)
			if N1 % 2 == 0:
				po1 = (1 - Fb(I_rest, vth, pon))
				po2 = fq * (1 - Fb(I_rest_2, vth, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth, poff))
				pcc = po1*po2 + (1-po1)*(1-po2)
			else:
				po1 = (1 - Fb(I_rest, vth - cnums, pon))
				po2 = fq * (1 - Fb(I_rest_2, vth - cnums, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth- cnums, poff))
				pcc = po1 * po2 + (1 - po1) * (1 - po2)
			p1 += prob_N1 * pcc
		p_final += prob_I0 * p1
	p_final = min(p_final, 1)
	return p_final


def pl_nonorgthogonal_sparse_coll(vth, M, n, fp, fq, I, I_, cnums, p_w, basic_I_list, basic_prop_list):
	# assume I is the overall number of 1s bits
	I_rest_2 = int(I_ * p_w)
	p_final = 0.
	for (I_rest, prob_I0) in basic_I_list:
		p1 = 0.
		for (N1, prob_N1) in basic_prop_list:
			pon = prob_on(fp, None, N1)
			poff = prob_off(fp, None, N1)
			if N1 % 2 == 0:
				po1 = (1 - Fb(I_rest, vth, poff))
				po2 = fq * (1 - Fb(I_rest_2, vth, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth, poff))
				pcc = po1 * po2 + (1 - po1) * (1 - po2)
			else:
				po1 = (1 - Fb(I_rest, vth - cnums, poff))
				po2 = fq * (1 - Fb(I_rest_2, vth - cnums, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth - cnums, poff))
				pcc = po1 * po2 + (1 - po1) * (1 - po2)
			p1 += prob_N1 * pcc
		p_final += prob_I0 * p1
	p_final = min(p_final, 1)
	return p_final

 
def overlap_nonorthogonal_sparse_coll(I, vth, M, n, m, p_rest, fq, comm_num, p_w=1., jumps=7):
	error_tols = 1e-8
	comm_avg = int(comm_num*p_w)
	I_tmp = [(idx, gamma_binomal(m - comm_num, idx, p_rest)) for (idx, n_I) in enumerate(range(2 * I))]
	q_tmp = [(idx, gamma_binomal(n, n_q, fq)) for (idx, n_q) in enumerate(range(int(3 * n * fq)))]
	I_prob = prob_shuffle(raw_list=I_tmp,jump=jumps,error_tol = error_tols)
	q_prob = prob_shuffle(raw_list=q_tmp,jump=3,error_tol = error_tols)

	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, int(fq * M * 2))]
	basic_prop_list = prob_shuffle(raw_list=basic_temp_list, jump=5, error_tol=error_tols)
	p2 = 0.
	for j0, prob_q in q_prob:
		n_q = j0
		p1 = 0.
		for _, item in enumerate(I_prob):
			i1, prob_I1 = item
			basic_I_list = [(x, binomal(i1, x, p_w)) for x in range(0, i1 + 1)]
			if len(basic_I_list) > 50:
				basic_I_list = prob_shuffle(basic_I_list, jump=3)
			p0 = 0.
			for _, item in enumerate(I_prob):
				i2, prob_I2 = item
				pa =  pg_nonorgthogonal_sparse_coll(vth, M, n, p_rest, fq, i1, i2, comm_avg, p_w, basic_I_list,basic_prop_list)
				pb =  pl_nonorgthogonal_sparse_coll(vth, M, n, p_rest, fq, i1, i2, comm_avg, p_w, basic_I_list,basic_prop_list)
				p_temp1 = prob_I2 * pa ** n_q * pb ** (n - n_q)
				p0 +=  p_temp1
			p1 += prob_I1 * p0
		# print(j0, p2)
		p2 += prob_q * p1
	p_final = p2
	return p_final




"""
some attempts
"""
# def overlap_nonorthogonal_sparse_coll_v2(I, vth, M, n, m, p_rest, fq, comm_num, p_w=1., jumps=7):
# 	error_tols = 1e-8
# 	comm_avg = int(comm_num * p_w)
# 	I_tmp = [(idx, gamma_binomal(m - comm_num, idx, p_rest * p_w)) for (idx, n_I) in enumerate(range(2 * I))]
# 	q_tmp = [(idx, gamma_binomal(n, n_q, fq)) for (idx, n_q) in enumerate(range(int(3 * n * fq)))]
# 	I_prob = prob_shuffle(raw_list=I_tmp,jump=jumps,error_tol = error_tols)
# 	q_prob = prob_shuffle(raw_list=q_tmp,jump=jumps,error_tol = error_tols)
#
# 	basic_temp_list = [(x, gamma_binomal(M, x, fq)) for x in range(0, int(fq * M * 2))]
# 	basic_prop_list = prob_shuffle(raw_list=basic_temp_list, jump=3, error_tol=error_tols)
# 	p2 = 0.
# 	for j0, prob_q in q_prob:
# 		n_q = j0
# 		p1 = 0.
# 		for _, item in enumerate(I_prob):
# 			i1, prob_I1 = item
# 			# calculate in advanced
# 			basic_I_list = [(x, binomal(i1, x, p_w)) for x in range(0, i1 + 1)]
# 			if len(basic_I_list) > 50:
# 				basic_I_list = prob_shuffle(basic_I_list, jump=3)
# 			p0 = 0.
# 			for _, item in enumerate(I_prob):
# 				i2, prob_I2 = item
# 				pa =  pg_nonorgthogonal_sparse_coll_v2(vth, M, n, p_rest, fq, i1, i2, comm_avg, p_w, basic_I_list,basic_prop_list)
# 				pb =  pl_nonorgthogonal_sparse_coll_v2(vth, M, n, p_rest, fq, i1, i2, comm_avg, p_w, basic_I_list,basic_prop_list)
# 				p_temp1 = prob_I2 * pa ** n_q * pb ** (n - n_q)
# 				p0 +=  p_temp1
# 			p1 += prob_I1 * p0
# 		# print(j0, p2)
# 		p2 += prob_q * p1
# 	p_final = p2
# 	return p_final


# def pg_nonorgthogonal_sparse_coll_v2(vth, M, n, fp,fq, I, I_, cnums, p_w, basic_I_list, basic_prop_list ):
# 	# assume I is the overall number of 1s bits
# 	I_rest_2 = I_
# 	p_final = 0.
# 	p1 = 0.
# 	for (N1, prob_N1) in basic_prop_list:
# 		pon = prob_on(fp, None, N1)
# 		poff = prob_off(fp, None, N1)
# 		if N1 % 2 == 0:
# 			po1 = (1 - Fb(I, vth, pon))
# 			po2 = fq * (1 - Fb(I_rest_2, vth, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth, poff))
# 			pcc = po1*po2 + (1-po1)*(1-po2)
# 		else:
# 			po1 = (1 - Fb(I, vth - cnums, pon))
# 			po2 = fq * (1 - Fb(I_rest_2, vth - cnums, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth- cnums, poff))
# 			pcc = po1 * po2 + (1 - po1) * (1 - po2)
# 		p1 += prob_N1 * pcc
# 		p_final +=  p1
# 	p_final = min(p_final, 1)
# 	return p_final
#
#
# def pl_nonorgthogonal_sparse_coll_v2(vth, M, n, fp, fq, I, I_, cnums, p_w, basic_I_list, basic_prop_list):
# 	# assume I is the overall number of 1s bits
# 	I_rest_2 = I_
# 	p_final = 0.
# 	p1 = 0.
# 	for (N1, prob_N1) in basic_prop_list:
# 		pon = prob_on(fp, None, N1)
# 		poff = prob_off(fp, None, N1)
# 		if N1 % 2 == 0:
# 			po1 = (1 - Fb(I, vth, poff))
# 			po2 = fq * (1 - Fb(I_rest_2, vth, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth, poff))
# 			pcc = po1*po2 + (1-po1)*(1-po2)
# 		else:
# 			po1 = (1 - Fb(I, vth - cnums, poff))
# 			po2 = fq * (1 - Fb(I_rest_2, vth - cnums, pon)) + (1 - fq) * (1 - Fb(I_rest_2, vth- cnums, poff))
# 			pcc = po1 * po2 + (1 - po1) * (1 - po2)
# 		p1 += prob_N1 * pcc
# 		p_final +=  p1
# 	p_final = min(p_final, 1)
# 	return p_final


def mask_sparse_hamming(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	error_tols = 1e-8
	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1) # supp(x)
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=len(q_temp)//50+1, error_tol=error_tols)
	p_final = 0.

	for I1, prob_i1 in I_prob:
		Im = mask_num
		Ir = I1 - mask_num
		if Ir > 0:
			Im_list = [(x, binomal(Im, x, p_w)) for x in range(Im + 1)]
			Ir_list = [(x, binomal(Ir, x, p_w)) for x in range(Ir + 1)]
			if len(Im_list) > 50:
				Im_list = prob_shuffle(Im_list, jump=2)
			if len(Ir_list) > 50:
				Ir_list = prob_shuffle(Ir_list, jump=2)
			p1 = 0.
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				for i2, prob_i2 in Im_list:
					for i3, prob_i3 in Ir_list:
						p_factor = prob_i2 * prob_i3
						p2, p3 = 0., 0.,
						for v in range(i3+1):
							if v <= vth:
								f_on = binomal(i3, v, p_on)
								f_off = binomal(i3, v, p_off)
								F_alpha1 = Fb(i2 , vth - v, p_on)
								F_alpha2 = Fb(i2,  vth - v, p_off)
								tmp0 = f_on * (1 - F_alpha1)
								tmp1 = f_off * (1 - F_alpha2)
								p2 += p_factor * tmp0
								p3 += p_factor * tmp1
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final

# def mask_hamming_simpler(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
# 	p_on = prob_on(fp * fq, 0, M - 1)
# 	p_off = prob_off(fp * fq, 0, M - 1)
# 	p_f1 = Fb(I, vth, p_on)
# 	p_f2 = Fb(I, vth, p_off)
# 	p_m1 = Fb(I-mask_num, vth, p_on)
# 	p_m2 = Fb(I-mask_num, vth, p_off)
# 	p_est = fq * (1 - p_f1) * p_m1 + (1 - fq) * (1 - p_f2) * p_m2
# 	return p_est

# @jit
# def mask_sparse_hamming_simpler(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
# 	p_on = prob_on(fp * fq, 0, M)
# 	p_off = prob_off(fp * fq, 0, M)
# 	p_f1 = Fb(I, vth, p_w*p_on) # probs. that the neuron fire after learning
# 	p_f2 = Fb(I, vth, p_w*p_off)
# 	p_m1 = Fb(I-mask_num, vth, p_w*p_on)
# 	p_m2 = Fb(I-mask_num, vth, p_w*p_off)
# 	p_est = fq * (1 - p_f1) * p_m1 + (1 - fq) * (1 - p_f2) * p_m2
# 	p_firing = fq * (1 - p_f1) * 1 + (1 - fq) * (1 - p_f2) * 1
# 	return p_est

def mask_sparse_hamming_firing(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	p_on = prob_on(fp * fq, 0, M)
	p_off = prob_off(fp * fq, 0, M)
	I_tmp = [ (idx , gamma_binomal(m , n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=1) # supp(x)
	p_final = 0.
	for I, prob_i1 in I_prob:
		p_f1 = Fb(I-mask_num, vth, p_w*p_on) # probs. that the neuron fire after learning
		p_f2 = Fb(I-mask_num, vth, p_w*p_off)
		p_firing = fq * (1 - p_f1) * 1 + (1 - fq) * (1 - p_f2) * 1
		p_final += prob_i1 * p_firing
	return p_final

def mask_sparse_hamming_simple(I, vth, M, m, n, fp, fq, fd, p_w,Q_prob=None,I_prob=None,jumps=1):
	"""
	HD between complete patterns and masking patterns
	Returns:
	"""
	if Q_prob is None:
		q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
		Q_prob = prob_shuffle(raw_list=q_temp, jump=jumps )

	if I_prob is None:
		I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps)

	p_final = 0.
	for I1, prob_i1 in I_prob:
		Im = int(I1 * fd)
		Ir = I1-Im
		p1 = 0.
		if Ir >= 0:
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				p2, p3 = 0., 0.,
				for v in range(Ir+1):
					if v <=  vth:
						f_on = gamma_binomal(Ir, v, p_w*p_on)
						f_off = gamma_binomal(Ir, v, p_w*p_off)
						F_alpha1 = Fb(Im , vth - v, p_w*p_on)
						F_alpha2 = Fb(Im,  vth - v, p_w*p_off)
						p2 +=   f_on * (1 - F_alpha1)
						p3 +=  f_off * (1 - F_alpha2)
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final

def mask_sparse_hamming_large(I, vth, M, m, n, fp, fq, fd, p_w,Q_prob=None,I_prob=None,jumps=1):
	"""
	HD between complete patterns and masking patterns
	Returns:
	"""

	if Q_prob is None:
		jumps_q = jumps
		q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
		Q_prob = prob_shuffle(raw_list=q_temp, jump=jumps )
		if len(Q_prob) > 100:
			jumps_q += 2
			Q_prob = prob_shuffle(raw_list=q_temp, jump=jumps_q)

	if I_prob is None:
		I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=3)

	p_final = 0.
	for I1, prob_i1 in I_prob:
		Im = int(I1 * fd)
		Ir = I1-Im
		p1 = 0.
		if Ir >= 0:
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				p2, p3 = 0., 0.,
				for v in range(Ir+1):
					if v <=  vth:
						f_on = gamma_binomal(Ir, v, p_w*p_on)
						f_off = gamma_binomal(Ir, v, p_w*p_off)
						F_alpha1 = Fb(Im , vth - v, p_w*p_on)
						F_alpha2 = Fb(Im,  vth - v, p_w*p_off)
						p2 +=   f_on * (1 - F_alpha1)
						p3 +=  f_off * (1 - F_alpha2)
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final


def mask_sparse_hamming_simple_jump(I, vth, M, m, n, fp, fq, mask_num, p_w,Q_prob=None,I_prob=None,jumps=5):
	"""
	HD between complete patterns and masking patterns
	"""
	jumps = min(jumps, 15)
	if Q_prob is None:
		q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
		Q_prob = prob_shuffle(raw_list=q_temp, jump=5)

	if I_prob is None:
		I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps*2+1)

	p_final = 0.
	for I1, prob_i1 in I_prob:
		Im, Ir = mask_num, I1 - mask_num
		p1 = 0.
		if Ir > 0:
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				p2, p3 = 0., 0.,
				for v in range(0,Ir+1,jumps):
					if v <= vth:
						f_on = gamma_binomal(Ir, v, p_w*p_on)
						f_off = gamma_binomal(Ir, v, p_w*p_off)
						F_alpha1 = Fb(Im , vth - v, p_w*p_on)
						F_alpha2 = Fb(Im,  vth - v, p_w*p_off)
						p2 +=   f_on * (1 - F_alpha1)
						p3 +=    f_off * (1 - F_alpha2)
				p1 += jumps * prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final

def mask_sparse_hamming_firing_v2(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=1):
	jumps = min(jumps, 15)
	I_tmp = [ (idx , gamma_binomal(m , n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)
	p_final = 0.
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))]  # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump= 3)

	for I, prob_i1 in I_prob:
		p1 = 0.
		for N1, prob_N1 in Q_prob:
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			p_f1 = Fb(I-mask_num, vth, p_w*p_on) # probs. that the neuron fire after learning
			p_f2 = Fb(I-mask_num, vth, p_w*p_off)
			p_firing = fq * (1 - p_f1)   + (1 - fq) * (1 - p_f2)
			p1 += prob_N1 * p_firing
		p_final += prob_i1 * p1
	return p_final


def mask_sparse_hamming_firing_v3(I, vth, M, m, n, fp, fq, fd, p_w,jumps=3):
	jumps = min(jumps, 15)
	I_tmp = [ (idx , gamma_binomal(m , n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)
	p_final = 0.
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))]  # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump= 3)

	for I, prob_i1 in I_prob:
		p1 = 0.
		mask_num = int(I*fd)
		for N1, prob_N1 in Q_prob:
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			p_f1 = Fb(I-mask_num, vth, p_w*p_on) # probs. that the neuron fire after learning
			p_f2 = Fb(I-mask_num, vth, p_w*p_off)
			p_firing = fq * (1 - p_f1)   + (1 - fq) * (1 - p_f2)
			p1 += prob_N1 * p_firing
		p_final += prob_i1 * p1
	return p_final

def mask_sparse_hamming_firing_large(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=1):
	jumps = min(jumps, 15)
	I_tmp = [ (idx , gamma_binomal(m , n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)
	p_final = 0.
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))]  # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump= 3)
	jumps = min(jumps, 15)
	if Q_prob is None:
		jumps_q = jumps
		q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
		Q_prob = prob_shuffle(raw_list=q_temp, jump=jumps )
		if len(Q_prob) > 100:
			jumps_q += 2
			Q_prob = prob_shuffle(raw_list=q_temp, jump=jumps_q)


	for I, prob_i1 in I_prob:
		p1 = 0.
		for N1, prob_N1 in Q_prob:
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			p_f1 = Fb(I-mask_num, vth, p_w*p_on) # probs. that the neuron fire after learning
			p_f2 = Fb(I-mask_num, vth, p_w*p_off)
			p_firing = fq * (1 - p_f1)   + (1 - fq) * (1 - p_f2)
			p1 += prob_N1 * p_firing
		p_final += prob_i1 * p1
	return p_final


def mask_sparse_hamming_scale(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	p_on = prob_on(fp * fq, 0, M)
	p_off = prob_off(fp * fq, 0, M)
	I_tmp = [ (idx , gamma_binomal(m , n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)
	p_final = 0.
	for I, prob_i1 in I_prob:
		if I-mask_num>=0:
			p_f1 = Fb(I, vth, p_w*p_on) # probs. that the neuron fire after learning
			p_f2 = Fb(I, vth, p_w*p_off)
			p_m1 = Fb(I-mask_num, vth, p_w*p_on)
			p_m2 = Fb(I-mask_num, vth, p_w*p_off)
			p_est = fq * (1 - p_f1) * p_m1 + (1 - fq) * (1 - p_f2) * p_m2
			p_firing = fq * (1 - p_f1) * 1 + (1 - fq) * (1 - p_f2) * 1
			p_final += prob_i1 * p_est
	return p_final

def mask_hamming_comm_simple(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=5):
	error_tols = 1e-11
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 3))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=1, error_tol=error_tols)
	p_final = 0.
	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)
	for I0, prob_i1 in I_prob:
		I1 = I0 + comm_num
		Im = mask_num
		Ir = I1 - mask_num
		p1 = 0.
		if Ir >= 0:
			common_ratio = comm_num / I1
			for N1, prob_N1 in Q_prob:
				# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
				if N1 % 2 == 1:
					p_on = prob_on(fp, None, N1) * (1 - common_ratio)
					p_off = prob_off(fp, None, N1)  * (1 - common_ratio) + 1 *  common_ratio
				else:
					p_on = prob_on(fp, None, N1) * (1 - common_ratio) + 1 * common_ratio
					p_off = prob_off(fp, None, N1) * (1 - common_ratio)
				p2, p3 = 0., 0.,
				for v in range(Ir+1):
					f_on = binomal(Ir, v, p_w * p_on)
					f_off = binomal(Ir, v, p_w * p_off)
					F_alpha1 = Fb(Im, vth - v, p_w * p_on)
					F_alpha2 = Fb(Im, vth - v, p_w * p_off)
					p2 += f_on * (1 - F_alpha1)
					p3 += f_off * (1 - F_alpha2)
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
		p_final += prob_i1 * p1
	return p_final




def mask_hamming_comm_simpler(I, vth, M, m, n, fp_rest, fq, mask_num, p_w, comm_num, jumps=5):
	# assembly for complete pattern firing, for masked pattern doesnot firing
	error_tols = 1e-8
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 3))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.

	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp_rest)) for (idx, n_I) in enumerate(range(int(m*fp_rest*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)
	for I0, prob_i1 in I_prob:
		I1 = I0 + comm_num
		Im = mask_num
		Ir = I1 - mask_num
		p1 = 0.
		if Ir >= 0:
			common_ratio = comm_num / I1
			for N1, prob_N1 in Q_prob:
				# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
				if N1 % 2 == 1:
					p_on = prob_on(fp_rest, None, N1) * (1 - common_ratio)
					p_off = prob_off(fp_rest, None, N1)  * (1 - common_ratio) + 1 *  common_ratio
				else:
					p_on = prob_on(fp_rest, None, N1) * (1 - common_ratio) + 1 * common_ratio
					p_off = prob_off(fp_rest, None, N1) * (1 - common_ratio)

				p2, p3 = 0., 0.,
				# pf1_mask = Fb(Ir , vth , p_w * p_on)
				# pf1_complete = Fb(I1, vth  , p_w * p_on)
				#
				# pf2_mask = Fb(Ir , vth , p_w * p_off)
				# pf2_complete = Fb(I1, vth  , p_w * p_off)
				#
				# p1 += prob_N1 * (fq * pf1_mask * (1-pf1_complete) + (1 - fq) * pf2_mask * (1-pf2_complete))

				# pf2 = Fb(Ir, vth - v, p_w * p_off)
				for v in range(Ir+1):
					if v <= vth:
						f_on = binomal(Ir, v, p_w*p_on)
						f_off = binomal(Ir, v, p_w*p_off)
						F_alpha1 = Fb(Im , vth - v, p_w*p_on)
						F_alpha2 = Fb(Im,  vth - v, p_w*p_off)
						p2 +=   f_on * (1 - F_alpha1)
						p3 +=   f_off * (1 - F_alpha2)
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final

 
# def mask_firing_rate_comm(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=2):
# 	error_tols = 1e-11
# 	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 3))] # Q(x)
# 	Q_prob = prob_shuffle(raw_list=q_temp, jump=1, error_tol=error_tols)
# 	p_final = 0.
# 	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
# 	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)
#
# 	c_tmp = [(idx, binomal(comm_num, n_w, p_w)) for (idx, n_w) in enumerate(range(comm_num+1))]
# 	c_prop =  prob_shuffle(raw_list=c_tmp, jump=jumps)
# 	for I0, prob_i1 in I_prob:
# 		I1 = I0
# 		p1 = 0.
# 		for N1, prob_N1 in Q_prob:
# 			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
# 			p_on = prob_on(fp, None, N1)
# 			p_off = prob_off(fp, None, N1)
# 			if N1 % 2 == 1:
# 				for c_w, prob_w in c_prop:
# 					p_f1 = Fb(I1, vth, p_w * p_on)  # probs. that the neuron fire after learning
# 					if c_w<vth:
# 						p_f2 = Fb(I1 , vth - c_w, p_w * p_off)
# 					else:
# 						p_f2 = 1
# 					p1 += prob_N1 * prob_w * (fq * (1 - p_f1) + (1 - fq) * (1 - p_f2))
# 			else:
# 				for c_w, prob_w in c_prop:
# 					if c_w <= vth:
# 						p_f1 = Fb(I1, vth-c_w, p_w * p_on)  # probs. that the neuron fire after learning
# 					else:
# 						p_f1 = 1.
# 					p_f2 = Fb(I1, vth, p_w * p_off)
# 					p1 += prob_N1 * prob_w * (fq * (1 - p_f1) + (1 - fq) * (1 - p_f2))
#
# 		p_final += prob_i1 * p1
# 	return p_final

def mask_firing_rate_comm_v2(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=2, q_w=1.):
	error_tols = 1e-8
	mask_ratio = mask_num/I
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.
	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)

	c_tmp = [ (idx , gamma_binomal(comm_num, n_w, p_w )) for (idx, n_w) in enumerate(range(int(comm_num)+1))]
	c_prop = c_tmp
	c_mask = int(comm_num * mask_ratio)
	mask_rest = mask_num - c_mask
	for I0, prob_i1 in I_prob:
		I1 = I0
		p1 = 0.
		for N1, prob_N1 in Q_prob:
			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			if N1 % 2 == 1:
				for c_w, prob_w in c_prop:
					c_w_ = c_w - int(c_mask*p_w)
					if c_w_ >= 0:
						p_f1_mask     = 1 - Fb(I1 - mask_rest, vth, p_w * p_on)
						p_f2_mask = 1 -  Fb(I1 - mask_rest, vth - c_w_, p_w * p_off)
						p1 += prob_N1 * prob_w * (fq * p_f1_mask  + (1 - fq) * p_f2_mask )
			else:
				for c_w, prob_w in c_prop:
					c_w_ = c_w - int(c_mask*p_w)
					if c_w_ >= 0:
						p_f2_mask     = 1 - Fb(I1-mask_rest, vth, p_w * p_off)  # probs. that the neuron fire after learning
						p_f1_mask = 1 - Fb(I1 - mask_rest, vth - c_w_, p_w * p_on)
						p1 += prob_N1 * prob_w * (fq * p_f1_mask + (1 - fq) * p_f2_mask )

		p_final += prob_i1 * p1
	return p_final

def mask_firing_rate_comm_v0(I, vth, M, m, n, fp_rest, fq,  p_w, comm_num, jumps=2 ):
	error_tols = 1e-8

	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)

	# here fp refers to the original fp
	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp_rest)) for (idx, n_I) in enumerate(range(int(m*fp_rest*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)
	c_prop = [ (idx , gamma_binomal(comm_num, n_w, p_w )) for (idx, n_w) in enumerate(range(int(comm_num)+1))]

	mask_rest = comm_num

	p_final = 0.
	for I1, prob_i1 in I_prob:

		p1 = 0.
		for c_w, prob_w in c_prop:
			c_w_ = c_w
			for N1, prob_N1 in Q_prob:
				# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
				p_on = prob_on(fp_rest, None, N1)
				p_off = prob_off(fp_rest, None, N1)
				if I1 -mask_rest*p_w>0:
					p_f1_mask1 = 1 - Fb(I1 , vth - c_w_, p_w * p_on)
					p_f1_mask2 = 1 - Fb(I1, vth, p_w * p_on)
					p_f2_mask1 = 1 - Fb(I1, vth  - c_w_, p_w * p_off)
					p_f2_mask2 = 1 - Fb(I1, vth, p_w * p_off)  #
					p1 += prob_N1 * prob_w * (fq * (p_f1_mask1+p_f1_mask2)/2 + (1 - fq) *  (p_f2_mask1+p_f2_mask2)/2)
					# print(p_final, prob_i1, p_f1_mask1, p_f2_mask1)

		p_final += prob_i1 * p1

	return p_final


def mask_firing_rate_comm_pattern(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=2):
	error_tols = 1e-11
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 3))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=1, error_tol=error_tols)
	p_final = 0.
	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=jumps) # supp(x)

	c_tmp = [(idx, binomal(comm_num, n_w, p_w)) for (idx, n_w) in enumerate(range(comm_num+1))]
	c_prop =  prob_shuffle(raw_list=c_tmp, jump=jumps)
	for I0, prob_i1 in I_prob:
		I1 = I0
		p1 = 0.
		for N1, prob_N1 in Q_prob:
			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			if N1 % 2 == 1:
				for c_w, prob_w in c_prop:
					p_f1 = Fb(I1, vth, p_w * p_on)  # probs. that the neuron fire after learning
					if c_w<vth:
						p_f2 = Fb(I1 , vth - c_w, p_w * p_off)
					else:
						p_f2 = 1
					p1 += prob_N1 * prob_w * (fq * (1 - p_f1) * p_f1 * 2 + (1 - fq) * (1 - p_f2) * p_f2 * 2)
			else:
				for c_w, prob_w in c_prop:
					if c_w <= vth:
						p_f1 = Fb(I1, vth-c_w, p_w * p_on)  # probs. that the neuron fire after learning
					else:
						p_f1 = 1.
					p_f2 = Fb(I1, vth, p_w * p_off)
					p1 += prob_N1 * prob_w * (fq * (1 - p_f1) * p_f1 * 2 + (1 - fq) * (1 - p_f2) * p_f2 * 2)

		p_final += prob_i1 * p1
	return p_final


def mask_hamming_comm_simpler(I, vth, M, m, n, fp_rest, fq, mask_num, p_w, comm_num, jumps=5):
	# assembly for complete pattern firing, for masked pattern doesnot firing
	error_tols = 1e-8
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 3))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.

	I_tmp = [ (idx , gamma_binomal(m-comm_num, n_I, fp_rest)) for (idx, n_I) in enumerate(range(int(m*fp_rest*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)
	for I0, prob_i1 in I_prob:
		I1 = I0 + comm_num
		Im = mask_num
		Ir = I1 - mask_num
		p1 = 0.
		if Ir >= 0:
			common_ratio = comm_num / I1
			for N1, prob_N1 in Q_prob:
				# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
				if N1 % 2 == 1:
					p_on = prob_on(fp_rest, None, N1) * (1 - common_ratio)
					p_off = prob_off(fp_rest, None, N1)  * (1 - common_ratio) + 1 *  common_ratio
				else:
					p_on = prob_on(fp_rest, None, N1) * (1 - common_ratio) + 1 * common_ratio
					p_off = prob_off(fp_rest, None, N1) * (1 - common_ratio)
				p2, p3 = 0., 0.,
				for v in range(Ir+1):
					if v <= vth:
						f_on = binomal(Ir, v, p_w*p_on)
						f_off = binomal(Ir, v, p_w*p_off)
						F_alpha1 = Fb(Im , vth - v, p_w*p_on)
						F_alpha2 = Fb(Im,  vth - v, p_w*p_off)
						p2 +=   f_on * (1 - F_alpha1)
						p3 +=   f_off * (1 - F_alpha2)
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final


def sub_mask_comm_firing_rate(vth, I1, c_mask, mask_rest,  c_prop, Q_prob, fp, fq, p_w ):
	p1 = 0.

	for N1, prob_N1 in Q_prob:
		# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
		p_on = prob_on(fp, None, N1)
		p_off = prob_off(fp, None, N1)
		if N1 % 2 == 1:
			for c_w, prob_w in c_prop:
				c_w_ = c_w - int(c_mask * p_w)
				if c_w_ >= 0:
					p_f1_mask = 1 - Fb(I1 - mask_rest, vth, p_w * p_on)
					p_f2_mask = 1- Fb(I1 - mask_rest, vth - c_w_, p_w * p_off)
					p1 += prob_N1 * prob_w * (fq * p_f1_mask + (1 - fq) * p_f2_mask)
		else:
			for c_w, prob_w in c_prop:
				c_w_ = c_w - int(c_mask * p_w)
				if c_w_ >= 0:
					p_f2_mask = 1 - Fb(I1 - mask_rest, vth, p_w * p_off)  # probs. that the neuron fire after learning
					p_f1_mask = 1 - Fb(I1 - mask_rest, vth - c_w_, p_w * p_on)
					p1 += prob_N1 * prob_w * (fq * p_f1_mask + (1 - fq) * p_f2_mask)

	return p1

def mask_hamming_comm_simpler_v0(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=5):
	# assume the fixed common bits and the rest bits are masked
	error_tols = 1e-8
	mask_ratio = mask_num/I
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.

	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)

	c_tmp = [ (idx , gamma_binomal(comm_num, n_w, p_w )) for (idx, n_w) in enumerate(range(int(comm_num)+1))]
	c_prop = c_tmp
	c_mask = int(comm_num * mask_ratio)
	mask_rest = mask_num - c_mask

	for I0, prob_i1 in I_prob:
		I1 = I0
		p1 = 0.
		for N1, prob_N1 in Q_prob:
			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)

			if N1 % 2 == 1:
				for c_w, prob_w in c_prop:
					c_w_ = c_w - int(c_mask*p_w)
					if c_w_ >= 0:
						# p_f1_mask     = Fb(I1-mask_num, vth, p_w * p_on)  # probs. that the neuron fire after learning
						p_f1_complete = Fb(I1,            vth, p_w * p_on)
						p_f1_mask     = Fb(I1 - mask_rest, vth, p_w * p_on)
						p_f2_complete = Fb(I1, vth - c_w, p_w * p_off)
						p_f2_mask = Fb(I1 - mask_rest, vth - c_w_, p_w * p_off)

						# p_f2_complete = Fb(I1, vth - c_mask, p_w * p_off)
						# p_f2_mask = Fb(I1 - mask_rest, vth - c_mask, p_w * p_off)
						p1 += prob_N1 * prob_w * (fq * p_f1_mask *(1-p_f1_complete) + (1 - fq) * p_f2_mask * (1 - p_f2_complete))
			else:
				for c_w, prob_w in c_prop:
					c_w_ = c_w - int(c_mask*p_w)
					if c_w_ >= 0:
						p_f2_complete = Fb(I1,           vth, p_w * p_off)
						p_f2_mask     = Fb(I1-mask_rest, vth, p_w * p_off)  # probs. that the neuron fire after learning

						p_f1_complete = Fb(I1, vth - c_w, p_w * p_on)
						p_f1_mask = Fb(I1 - mask_rest, vth - c_w_, p_w * p_on)
						p1 += prob_N1 * prob_w * (fq * p_f1_mask *(1-p_f1_complete) + (1 - fq) * p_f2_mask * (1 - p_f2_complete))

		p_final += prob_i1 * p1
	return p_final


def mask_hamming_comm_simpler_v1(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=5):
	# assume the fixed common bits and the rest bits are masked
	error_tols = 1e-8
	fd = mask_ratio = mask_num/I
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.

	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)

	c_tmp = [ (idx , gamma_binomal(comm_num, n_w, p_w )) for (idx, n_w) in enumerate(range(int(comm_num)+1))]
	c_prop = c_tmp
	c_mask = int(comm_num * mask_ratio)

	for I0, prob_i1 in I_prob:
		I  = I0 + comm_num
		fd = mask_num / I
		p1 = 0.
		mask_rest = mask_num - c_mask
		for N1, prob_N1 in Q_prob:
			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)
			c_w = comm_num * p_w
			if N1 % 2 == 1:
				p_f1_complete = Fb(I - comm_num, vth, p_w * p_on)
				p_f1_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth, p_w * p_on)
				p_f2_complete = Fb(I - comm_num, vth - c_w, p_w * p_off)
				p_f2_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth - c_w * (1-fd), p_w * p_off)

			else:
				p_f1_complete = Fb(I-comm_num,                vth - c_w,          p_w * p_on)
				p_f1_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth - c_w * (1-fd), p_w * p_on)

				p_f2_complete = Fb(I-comm_num,                vth, p_w * p_off)
				p_f2_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth, p_w * p_off)  # probs. that the neuron fire after learning
			p1 += prob_N1 *  (
						fq * p_f1_mask * (1 - p_f1_complete) + (1 - fq) * p_f2_mask * (1 - p_f2_complete))
		p_final += prob_i1 * p1
	return p_final


def mask_hamming_comm_simpler_v2(I, vth, M, m, n, fp, fq, mask_num, p_w, comm_num, jumps=5):
	# assume the fixed common bits and the rest bits are masked
	error_tols = 1e-8
	fd = mask_ratio = mask_num/I
	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)
	p_final = 0.

	I_tmp = [ (idx , gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m*fp*4)))]
	I_prob = prob_shuffle(raw_list=I_tmp, jump=3) # supp(x)

	c_tmp = [ (idx , gamma_binomal(comm_num, n_w, p_w )) for (idx, n_w) in enumerate(range(int(comm_num)+1))]
	c_prop = c_tmp
	c_mask = int(comm_num * mask_ratio)

	for I0, prob_i1 in I_prob:
		I  = I0 + comm_num
		fd = mask_num / I
		p1 = 0.
		mask_rest = mask_num - c_mask
		for N1, prob_N1 in Q_prob:
			# according to whether the ca1 neuron get plateaus for xk during learning, we divide the estimate into pon and poff
			p_on = prob_on(fp, None, N1)
			p_off = prob_off(fp, None, N1)

			for c_w, prob_c in c_prop:

				if N1 % 2 == 1:
					p_f1_complete = Fb(I - comm_num, vth, p_w * p_on)
					p_f1_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth, p_w * p_on)
					p_f2_complete = Fb(I - comm_num, vth - c_w, p_w * p_off)
					p_f2_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth - c_w * (1-fd), p_w * p_off)

				else:
					p_f1_complete = Fb(I-comm_num,                vth - c_w,          p_w * p_on)
					p_f1_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth - c_w * (1-fd), p_w * p_on)

					p_f2_complete = Fb(I-comm_num,                vth, p_w * p_off)
					p_f2_mask     = Fb((I - comm_num) - (mask_num-comm_num * fd), vth, p_w * p_off)  # probs. that the neuron fire after learning
			p1 += prob_N1 *  prob_c * (
						fq * p_f1_mask * (1 - p_f1_complete) + (1 - fq) * p_f2_mask * (1 - p_f2_complete))
		p_final += prob_i1 * p1
	return p_final


def mask_purb_hamming_simple(I, vth, M, m, n, fp, fq, mask_num, p_w,jumps=5):
	error_tols = 1e-8

	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=5, error_tol=error_tols)
	p_final = 0.
	I_prob = [(I, 1.)]

	# fm = mask_num/I
	mask_num_list = [(x, gamma_binomal(mask_num, x, fp)) for x in range(mask_num+1)]
	for I1, prob_i1 in I_prob:

		p0 = 0.
		for comm_bit, prob_c in mask_num_list:
			I_com = comm_bit
			I_res = I1 - comm_bit
			p1 = 0.
			if I_com >= 0:
				for N1, prob_N1 in Q_prob:
					p_on = prob_on(fp, None, N1)
					p_off = prob_off(fp, None, N1)
					p2, p3 = 0., 0.,
					for v in range(I_com+1):
						f_on = binomal(I_com, v, p_w*p_on)
						f_off = binomal(I_com, v, p_w*p_off)
						F_alpha1 = Fb(I_res , vth - v, p_w * p_on)


						F_alpha2 = Fb(I_res,  vth - v, p_w * p_off)
						F_beta2 = Fb(I_res, vth - v, p_w * p_off )
						F_beta1 = Fb(I_res, vth - v, p_w * p_off)

						tmp0 =  ((1 - F_alpha1) * F_beta1 + F_alpha1 * (1 - F_beta1))
						tmp1 =  ((1 - F_alpha2) * F_beta2 + F_alpha2 * (1 - F_beta2))
						p2 +=   f_on * tmp0
						p3 +=   f_off * tmp1
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p0 += prob_c * p1
		p_final += prob_i1 * p0
	return p_final



def mask_purb_hamming_simpler(I, vth, M, m, n, fp, fq, mask_num, p_w,fixed_size=True, jumps=5):
	error_tols = 1e-8

	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=5, error_tol=error_tols)
	p_final = 0.

	if fixed_size is True:
		I_prob = [(I,1.)]
	else:
		I_tmp = [(idx, gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m * fp * 4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1)

	for I1, prob_i1 in I_prob:
		I_res = mask_num
		I_comm = I1 - mask_num
		p1 = 0.
		if I_comm > 0:
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				p2, p3 = 0., 0.,
				for v in range(I_comm+1):
					if v <= vth:
						f_on = binomal(I_comm, v, p_w*p_on)
						f_off = binomal(I_comm, v, p_w*p_off)
						F_alpha1 = Fb(I_res , vth - v, p_w*p_on)
						F_alpha2 = Fb(I_res,  vth - v, p_w*p_off)
						F_beta2 = Fb(I_res, vth - v, p_w * p_off )
						F_beta1 = Fb(I_res, vth - v, p_w * p_off)

						tmp0 =  ((1 - F_alpha1) * F_beta1 + F_alpha1 * (1 - F_beta1))
						tmp1 =  ((1 - F_alpha2) * F_beta2 + F_alpha2 * (1 - F_beta2))
						p2 +=   f_on * tmp0
						p3 +=   f_off * tmp1
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final




def mask_purb_hamming_random(I, vth, M, m, n, fp, fq, mask_num_all, p_w,fixed_size=True, jumps=3):
	error_tols = 1e-8

	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=3, error_tol=error_tols)


	if fixed_size is True:
		I_prob = [(I,1.)]
	else:
		I_tmp = [(idx, gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m * fp * 4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1)

	p_final_= 0.
	for I1, prob_i1 in I_prob:
		p_final = 0.
		mask_num_list = [(n_mask, binomal(I1, n_mask, mask_num_all/m)) for (idx, n_mask) in
						 enumerate(range(I1 + 1))]
		if len(mask_num_list) > 20:mask_num_list=prob_shuffle(raw_list=mask_num_list, jump=2)
		for mask_num, prob_mask in mask_num_list:
			I_res = mask_num_all - mask_num
			I_comm = I1 - mask_num
			p1 = 0.
			if I_comm >= 0 and prob_mask > 1e-9:
				for N1, prob_N1 in Q_prob:
					p_on = prob_on(fp, None, N1)
					p_off = prob_off(fp, None, N1)
					p2, p3 = 0., 0.,
					for v in range(I_comm+1):
						if v <= vth:
							f_on = binomal(I_comm, v, p_w*p_on)
							f_off = binomal(I_comm, v, p_w*p_off)
							F_alpha1 = Fb(I_res , vth - v, p_w*p_on)
							F_alpha2 = Fb(I_res,  vth - v, p_w*p_off)
							F_beta2 = Fb(I_res, vth - v, p_w * p_off )
							F_beta1 = Fb(I_res, vth - v, p_w * p_off)

							tmp0 =  ((1 - F_alpha1) * F_beta1 + F_alpha1 * (1 - F_beta1))
							tmp1 =  ((1 - F_alpha2) * F_beta2 + F_alpha2 * (1 - F_beta2))
							p2 +=   f_on * tmp0
							p3 +=   f_off * tmp1
					p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
				p_final += prob_mask * p1
		p_final_ += prob_i1 * p_final
	return p_final_



def mask_purb_hamming_simpler_v2(I, vth, M, m, n, fp, fq, mask_num, p_w,fixed_size=True, jumps=5):
	error_tols = 1e-8

	q_temp = [(x, gamma_binomal(M, x, fq)) for x in range(int(fq * M * 2))] # Q(x)
	Q_prob = prob_shuffle(raw_list=q_temp, jump=5, error_tol=error_tols)
	p_final = 0.

	if fixed_size is True:
		I_prob = [(I,1.)]
	else:
		I_tmp = [(idx, gamma_binomal(m, n_I, fp)) for (idx, n_I) in enumerate(range(int(m * fp * 4)))]
		I_prob = prob_shuffle(raw_list=I_tmp, jump=len(I_tmp)//50+1)

	for I1, prob_i1 in I_prob:
		I_res = int(mask_num * fp)
		I_comm = I1 - mask_num
		p1 = 0.
		if I_comm > 0:
			for N1, prob_N1 in Q_prob:
				p_on = prob_on(fp, None, N1)
				p_off = prob_off(fp, None, N1)
				p2, p3 = 0., 0.,
				for v in range(I_comm+1):
					if v <= vth:
						f_on = binomal(I_comm, v, p_w*p_on)
						f_off = binomal(I_comm, v, p_w*p_off)
						F_alpha1 = Fb(I_res , vth - v, p_w*p_on)
						F_alpha2 = Fb(I_res,  vth - v, p_w*p_off)
						F_beta2 = Fb(I_res, vth - v, p_w * p_off )
						F_beta1 = Fb(I_res, vth - v, p_w * p_off)

						tmp0 =  ((1 - F_alpha1) * F_beta1 + F_alpha1 * (1 - F_beta1))
						tmp1 =  ((1 - F_alpha2) * F_beta2 + F_alpha2 * (1 - F_beta2))
						p2 +=   f_on * tmp0
						p3 +=   f_off * tmp1
				p1 += prob_N1 * (fq * p2 + (1 - fq) * p3)
			p_final += prob_i1 * p1
	return p_final
