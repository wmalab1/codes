import numpy as np

def calc_E_D(d_per_bits, R_case, C_case, num_bits_case, Pout, num_pkt, D_STEP):
    max_pkt = num_pkt
    E_D = 0
    D = np.zeros([num_bits_case.shape[0], 1])  # 300x1

    for success_pkt in range(num_pkt, 0, -1):

        # 01.success_pkt에 따라 받은 총 비트수 계산(ex. r1+r2)
        total_bits = np.sum(num_bits_case[:, 0:success_pkt], axis=1)

#         small_v_idx = np.floor(total_bits / D_STEP)
#         big_v_idx = np.ceil(total_bits / D_STEP)
#
#         small_v = small_v_idx * D_STEP
#         big_v = big_v_idx * D_STEP
#
#         # for a in range(num_bits_case.shape[0]):
#         #     if (small_v[a] != big_v[a]):
#         #         w1 = (big_v[a] - total_bits[a]) / (big_v[a] - small_v[a])
#         #         w2 = (total_bits[a] - small_v[a]) / (big_v[a] - small_v[a])
#         #
#         #         D[a, 0] = w1 * d_per_bits[int(small_v_idx[a]), 1] + w2 * d_per_bits[int(big_v_idx[a]), 1]
#         #     else:
#         #         D[a, 0] = d_per_bits[int(small_v_idx[a]), 1]
#
# #############################################
#         #2020.01.05
#         mask_same = small_v == big_v
#         mask_dif = small_v != big_v
#
#         # if (small_v[a] != big_v[a])
#         w1 = (big_v[mask_dif] - total_bits[mask_dif]) / (big_v[mask_dif] - small_v[mask_dif])
#         w2 = (total_bits[mask_dif] - small_v[mask_dif]) / (big_v[mask_dif] - small_v[mask_dif])
#
#         # aaa = small_v_idx[mask_dif]
#         # aaaaa = small_v_idx[mask_dif].astype(int)
#         # aaaaaaaa = d_per_bits[int(small_v_idx[mask_dif]), 1]
#         # temp = w1 * d_per_bits[small_v_idx[mask_dif].astype(int), 1] + w2 * d_per_bits[big_v_idx[mask_dif].astype(int), 1]
#
#         # D[mask_dif, 0] = w1 * d_per_bits[int(small_v_idx[mask_dif]), 1] + w2 * d_per_bits[int(big_v_idx[mask_dif]), 1]
#         D[mask_dif, 0] = w1 * d_per_bits[small_v_idx[mask_dif].astype(int), 1] + w2 * d_per_bits[big_v_idx[mask_dif].astype(int), 1]
#
#         # if (small_v[a] == big_v[a])
#         D[mask_same, 0] = d_per_bits[small_v_idx[mask_same].astype(int), 1]
# ###################################################
#
#
#         if (success_pkt == max_pkt):
#             value = D
#         else:
#             value = D * Pout[:, success_pkt].reshape([num_bits_case.shape[0],1])

        # 2020.01.05
        D_idx1 = (total_bits / D_STEP).astype('int32')
        D_idx2 = D_idx1 + 1

        small_v = (D_idx1 * D_STEP).astype('float64')
        big_v = (D_idx2 * D_STEP).astype('float64')

        w1 = (big_v - total_bits) / (big_v - small_v)
        w2 = (total_bits - small_v) / (big_v - small_v)

        if (success_pkt == max_pkt):
            mask = total_bits < (2 ** 11) * num_pkt
            D[mask, 0] = w1[mask] * d_per_bits[D_idx1[mask], 1] + w2[mask] * d_per_bits[D_idx2[mask], 1]
            D[~mask, 0] = d_per_bits[D_idx1[~mask], 1]

            # for a in range(total_bits.shape[0]):
            #     if(total_bits[a] < (2**11) * num_pkt):
            #         D[a,0] = w1[a] * d_per_bits[D_idx1[a], 1] + w2[a] * d_per_bits[D_idx2[a], 1]
            #     else:
            #         D[a,0] = d_per_bits[D_idx1[a], 1]
            value = D
        else:
            D[:, 0] = w1 * d_per_bits[D_idx1, 1] + w2 * d_per_bits[D_idx2, 1]
            value = D * Pout[:, success_pkt].reshape([num_bits_case.shape[0], 1])

        E_D = (E_D + value) * (1 - Pout[:, success_pkt - 1].reshape([num_bits_case.shape[0],1]))

    f_0 = d_per_bits[0, 1]
    E_D = E_D + f_0 * Pout[:, 0].reshape([num_bits_case.shape[0],1])

    opt_alpha_idx = np.argmin(E_D)
    opt_E_D = E_D[opt_alpha_idx]
    opt_R = R_case[opt_alpha_idx, :]
    opt_C = C_case[opt_alpha_idx, :]

    return opt_E_D, opt_R, opt_C, opt_alpha_idx