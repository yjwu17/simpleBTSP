import copy

def polar_data(x):
    y = copy.deepcopy(x)
    y[y<=0] = -1
    return y

def reverse_polar_data(x):
    y = copy.deepcopy(x)
    y[y<0] = 0
    return y


def prob_shuffle(raw_list,jump=None,error_tol = 1e-1):
    p_list, p_sum = [], 0
    app_err = 1
    if jump is None:
        jump = len(raw_list) // 100 + 1

    while app_err > error_tol:
        if len(p_list) > 0: p_list, p_sum = [], 0
        for idx, prob in raw_list:
            if idx % jump == 0 and prob > 1e-10:
                p_list.append([idx, prob * jump])
                p_sum += prob * jump
        app_err = abs(p_sum - 1)
        if jump <= 1:
            break
        jump = jump - 1
    return p_list

def memory_size(x):
	print('memory size',x.element_size() * x.nelement()/1e9)


# from sparse_tools import *
# def overlap_nonorthogonal_sparse_v2(I, vth, M, n, m, p, fq, comm_num, p_w=1.,iters=1):
#     q = fq
#     I_tmp = [(idx, gamma_binomal(m - comm_num, idx, p)) for (idx, n_I) in enumerate(range(comm_num, 4 * I))]
#     q_tmp = [(idx, gamma_binomal(n, n_q, q)) for (idx, n_q) in enumerate(range(int(3 * n * fq)))]
#     basic_commm_list = [(x, binomal(comm_num, x, p_w)) for x in range(0, comm_num + 1)]
#     if len(basic_commm_list) > 50: basic_commm_list = prob_shuffle(basic_commm_list, jump=2)
#
#     I_prob, I_sum = [], []
#
#     p0 = 0.
#     for (cnums, prob_c1) in basic_commm_list:
#         p1 = 0.
#         pg_list = [(i, pg_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], cnums, p_w)) for (i, x) in
#                    enumerate(I_prob)]
#         pl_list = [(i, pl_non_orgthogonal_sparse(vth, M, n, p, fq, x[0], cnums, p_w)) for (i, x) in
#                    enumerate(I_prob)]
#         p2 = 0.
#         for j0, prob_q in q_prob:
#             n_q = j0
#             p0, p1 = 0., 0.
#             for j1, item in enumerate(I_prob):
#                 i1, prob_I1 = item
#                 pg1 = pg_list[j1][1]
#                 pl1 = pl_list[j1][1]
#                 p_o1 = q * pg1 + (1 - q) * (1 - pl1)
#                 for j2, item in enumerate(I_prob):
#                     i2, prob_I2 = item
#                     prob_factor = prob_I1 * prob_I2
#                     if prob_factor * prob_q > 1e-13:
#                         pg2 = pg_list[j2][1]
#                         pl2 = pl_list[j2][1]
#                         pa = pg2 * p_o1 + (1 - pg2) * (1 - p_o1)
#                         pb = (1 - pl2) * p_o1 + pl2 * (1 - p_o1)
#                         p_temp1 = prob_factor * pa ** n_q * pb ** (n - n_q)
#                         p0 += p_temp1
#
#             p2 += prob_q * p0
#         p3 += prob_c1 * p2
#
#     p_final = p3
#     return p_final
