import numpy as np

def calc_E_D_prime(a, num_bits, Pout, num_pkt, R, C):
    opt_E_D_temp = 1
    R_opt = np.zeros([1, num_pkt])
    C_opt = np.zeros([1, num_pkt])

    for pkt_idx in range(num_pkt,0,-1):
        f_x = 2**(-a*num_bits)
        E_D_temp = Pout + f_x*(1-Pout)*opt_E_D_temp

        idx = np.argmin(E_D_temp)
        opt_R_idx = idx % len(R)
        opt_C_idx = int(idx / len(R))
        opt_E_D_temp = E_D_temp[opt_C_idx, opt_R_idx]

        R_opt[0, pkt_idx - 1] = R[opt_R_idx]
        C_opt[0, pkt_idx - 1] = C[opt_C_idx]

    return R_opt, C_opt