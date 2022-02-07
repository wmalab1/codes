import glob
import os
import sys

import numpy as np
import csv
import time

from calc_E_D_prime import calc_E_D_prime
from calc_E_D import calc_E_D

#-------------------------------------------
# 01. parameter 설정
#-------------------------------------------
total_sample = 1000

# num_sample = 1
# num_sample = 10
# num_sample = 100
num_sample = 1000

num_iter = int(total_sample/num_sample)

R = np.array([1,2,3,4]) #R.shape[0] = 4, R.shape[1] = 존재x
C = np.array([1,4/3,2]).reshape([-1,1]) #C.shape[0] = 3, C.shape[1]= 1
# C = np.array([3,4,6]).reshape([-1,1]) #C.shape[0] = 3, C.shape[1]= 1

R_max = np.max(R)
C_max = np.max(C)


alpha_step = 1
# alpha_step = 0.5
# alpha_step = 0.1
alpha = np.arange(0, 30 + alpha_step, alpha_step)


# snr_dB = np.array([0])
# snr_dB = np.arange(-10,61,1)
snr_dB = np.arange(0,61,5)
# snr_dB = np.arange(0,11,5)
snr = 10**(snr_dB/10)

# K = np.array([0])
K = np.array([0,4,8,12])


image_size = 512**2 * 1
num_pkt = 128
max_pkt_size = image_size / num_pkt

D_STEP = 256
# D_STEP = 2**13

# MIMO
num_bits = (max_pkt_size/512**2)*(C/C_max)*(R/R_max)

input_size = int(image_size/D_STEP + 1) #513
# d_per_bits = np.zeros([num_sample, input_size, 2])
d_per_bits = np.zeros([total_sample, input_size, 2])

step = np.arange(0,int(image_size/64) + 1, int(D_STEP/64))   #256 간격
# step = np.arange(0,int(image_size/64) + 1, 8)   #512 간격일 때
# step = np.arange(0,int(image_size/64) + 1, 1) #64 간격일 때


opt_R_per_alpha = np.zeros([alpha.shape[0],num_pkt])
opt_C_per_alpha = np.zeros([alpha.shape[0],num_pkt])
# Pout_per_alpha = np.zeros([alpha.shape[0],num_pkt])

#E_D_opt, R_opt, C_opt, alpha_opt: 모든 sample에 대한 값 저장
E_D_opt = np.zeros([snr_dB.shape[0],num_sample])
R_opt = np.zeros([snr_dB.shape[0],num_sample,num_pkt])
C_opt = np.zeros([snr_dB.shape[0],num_sample,num_pkt])
alpha_opt = np.zeros([snr_dB.shape[0],num_sample])

f_time = open('tcom_time.dat', 'w')

#=================================================================
# D-R curve 파일 읽어서 배열로 저장
#=================================================================
i = 0
input_path = sys.argv[0]
input_path = input_path[:-7]
for input_file in glob.glob('distortion_test_img_new\distort_a*'):
    if (i+1 <= total_sample):
        with open(input_file, 'r') as f:
            rdr = csv.reader(f, delimiter='\t')
            temp = np.array(list(rdr), dtype=np.float64)
            d_per_bits[i,:,:] = temp[step, :]
            i = i + 1


#=================================================================
# OSTBC, DBLAST, VBLAST일 때 rician factor, snr에 따른 Pout 읽어와서 4차원 배열에 저장
# rician factor = 변수 K에 해당
# snr range: -10, -9, ..., 60 dB
#=================================================================
Pout_all = np.zeros([K.shape[0], C.shape[0], 60-(-10)+1, R.shape[0]+1])   #3x71x5
Pout_all[:,:,:,0] = np.arange(-10,61,1)

file_list = ['OSTBC', 'DBLAST', 'VBLAST']

for k in range(K.shape[0]):
    for i in range(len(file_list)):
        for r in range(R.shape[0]):
            filename = 'outage_prob/outage_prob_rician/' + file_list[i] + '_K' + str(K[k]).zfill(2) + '_R' + str(R[r]) + '_2x2.dat'
            with open(filename, 'r') as f:
                rdr = csv.reader(f, delimiter='\t')
                temp = np.array(list(rdr), dtype=np.float64)
                data = temp.reshape([1, 71, 2])
                Pout_all[k,i,:,r+1] = data[0,:,1]  # R값에 따른 Pout만을 Pout_DB에 넣어줌.


#-------------------------------------------------------------------------------------------------
# 02. rician factor, snr에 따른 optimal E(D), R값 계산
# 02-1. f(x) = 2^-ax. alpha값에 따른 E'(D)와 optimal R, C 계산
# 02-2. f(x) = rate-distort 함수. 위에서 계산한 optimal R, C set에 따른 최종의 E(D),optimal R,C 도출
#-------------------------------------------------------------------------------------------------

# 02-1. f(x) = 2^-ax일 때, alpha값에 따른 E'(D)와 optimal R 계산
for k in range(K.shape[0]):
    for i in range(snr_dB.shape[0]):
        print('snr_dB: %d' % snr_dB[i])
        for j in range(num_iter):
            input = d_per_bits[j*num_sample:(j+1)*num_sample,:,:]

            start_sec = time.time()

            # 한 snr과 한 rician factor에 대한 Pout
            snr_idx = np.where(Pout_all[0, 0, :, 0] == snr_dB[i])
            Pout = Pout_all[k, :,snr_idx,1:].reshape([C.shape[0],R.shape[0]])
            # Pout = Pout_all_snr_stbc[:,snr_dB[i]+10,1:].reshape([C.shape[0],R.shape[0]]) -> 더 오래 걸림
            Pout = np.squeeze(Pout_all[k, :, snr_idx, 1:])

            for a in range(alpha.shape[0]):
                opt_R_per_alpha[a, :], opt_C_per_alpha[a, :] = calc_E_D_prime(alpha[a], num_bits, Pout, num_pkt, R, C)

            # 02-2. f(x) = rate-distort 함수. 위에서 계산한 optimal R set에 따른 최종 optimal E(D),optimal R 도출
            num_bits_case = max_pkt_size * (opt_R_per_alpha / R_max) * (opt_C_per_alpha / C_max)

            # 각 alpha의 R,C 가지고 Pout 계산.
            opt_R_idx = np.where(opt_R_per_alpha == 1, 0, np.where(opt_R_per_alpha == 2, 1, np.where(opt_R_per_alpha == 3, 2, 3)))
            opt_C_idx = np.where(opt_C_per_alpha == 1, 0, np.where(opt_C_per_alpha == 4 / 3, 1, 2))
            Pout_idx = np.dstack([opt_C_idx, opt_R_idx])
            Pout_per_alpha = Pout[Pout_idx[:,:,0], Pout_idx[:,:,1]]     # 31x128

            # for a in range(alpha.shape[0]):
            #     for p in range(num_pkt):
            #         Pout_per_alpha[a, p] = Pout[opt_C_idx[a, p], opt_R_idx[a, p]]

            for n in range(num_sample):
                E_D_opt[i,n], R_opt[i,n,:], C_opt[i, n, :], opt_alpha_idx = calc_E_D(input[n, :, :], opt_R_per_alpha, opt_C_per_alpha, num_bits_case, Pout_per_alpha, num_pkt, D_STEP)
                alpha_opt[i,n] = alpha[opt_alpha_idx]

            total_time = time.time() - start_sec


            f_time.write('%d\t%02d\t%02d\t%d\t%20.20g\n' % (j, snr_dB[i], K[k], num_sample, total_time))

            # print(total_time)

            filename_D = 'D_tcom_' + str(num_pkt) + 'pkt_reference_snr' + str(format(snr_dB[i], "02d")) + '_K' + str(format(K[k], "02d")) +'_R=[1 2 3 4].dat'
            filename_R_C = 'R_C_tcom_' + str(num_pkt) + 'pkt_reference_snr' + str(format(snr_dB[i], "02d")) + '_K' + str(format(K[k], "02d")) + '_R=[1 2 3 4].dat'

            with open(filename_D, 'w') as f, open(filename_R_C, 'w') as f1:
                for n in range(num_sample):
                    f.write('%d\t%d\t%g\n' % (snr_dB[i], n+1, E_D_opt[i, n]))
                    f1.write('%d\t%d' % (snr_dB[i], n+1))

                    for a in range(num_pkt):
                        f1.write('\t%g' % (R_opt[i,n,a]))

                    for a in range(num_pkt):
                        f1.write('\t%g' % (C_opt[i,n,a]))

                    f1.write('\n')

f_time.close()
# print("===========실행시간: %1.5f==========="%(time.time() - start_sec))

# for i in range(snr_dB.shape[0]):
#     filename = 'D_tcom_' + str(num_pkt) + 'pkt_reference_snr' + str(format(snr_dB[i],"02d")) + '_R=[1 2 3 4].dat'
#     with open(filename, 'w') as f:
#         for n in range(num_sample):
#             f.write('%d\t%d\t%g\n' %(snr_dB[i], n, E_D_opt[i,n]))


