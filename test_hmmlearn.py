from __future__ import print_function

import datetime
import time

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM


print(__doc__)

###############################################################################

test_close_v_list1 = []
test_volume_list_diff = []
test_close_v_list_real = []
test_fn_diff1 = 'test_diff_h1.txt'
test_fn_vol_diff = 'test_vol_h_diff.txt'
test_fn_diff_real = 'test_diff_real.txt'

with open(test_fn_diff1) as fp:
    for line in fp:
        try:
            test_close_v_list1.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise
with open(test_fn_vol_diff) as fp:
    for line in fp:
        try:
            test_volume_list_diff.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise
with open(test_fn_diff_real) as fp:
    for line in fp:
        try:
            test_close_v_list_real.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise

test_close_v1 = np.array(test_close_v_list1)
test_volume_diff = np.array(test_volume_list_diff)
test_close_v_real = np.array(test_close_v_list_real)
X = np.column_stack([test_close_v1, test_volume_diff])
###############################################################################
startprob = np.array([  2.85079576e-137 ,  1.99166192e-051 ,  0.00000000e+000,   0.00000000e+000,
   1.00000000e+000,   1.40697527e-267,   1.02075412e-306,   6.80366548e-108,
   4.30776170e-250,   2.66079890e-222])

transmat = np.array([[  5.78278667e-001,   2.52093132e-002,   1.15182039e-001 ,  9.43762871e-003,
    5.08913757e-003,   2.25571530e-002,   1.35485802e-021,   2.38910619e-001,
    5.32940157e-003,   6.04157984e-006],
  [  5.04798378e-006,   4.50558166e-001 ,  1.27887652e-027 ,  8.49005306e-103,
    4.20032545e-001,   2.85122984e-006  , 1.43763182e-005 ,  7.22447023e-002,
    5.71423112e-002,   2.14935143e-012],
  [  3.12591427e-002,   9.26647818e-003,   7.95660460e-001,   1.70184579e-033,
    5.34592355e-002 ,  4.59863125e-003,   9.45769344e-017, 9.39424208e-002,
    4.00582749e-009 ,  1.18136271e-002],
  [  1.71633304e-001,   1.69996652e-037,   9.53116074e-002,   4.78465124e-001,
    2.92351691e-064, 1.73162195e-001,   8.14277689e-002,   4.21606994e-010,
    6.29884114e-066, 2.15840593e-059],
 [  6.17678812e-002,   1.01522076e-001,   3.59144178e-002,   7.37248591e-064,
    5.16186692e-001,  6.51091942e-008,   6.25814147e-003,   4.75387740e-006,
    2.75338946e-002, 2.50812078e-001],
 [  8.49771818e-020,   1.23523369e-010,   4.29437332e-057,   4.85066112e-003,
    2.00781138e-029,   6.70502989e-001,  6.86481818e-002,  2.55998168e-001,
    1.71304618e-016,   1.33630198e-063],
 [  4.29917221e-108,  4.86007734e-002,  1.41895646e-155,   4.01006267e-086,
    3.58005908e-062,  3.29914463e-001,  4.55352716e-001,  4.12031864e-044,
    1.66132048e-001,  8.75715469e-143],
 [  3.64056400e-001,   1.83134994e-002,   2.77041618e-006,   2.92040879e-008,
    8.76130799e-003,  9.84901738e-002,   2.79692149e-003,   5.07577977e-001,
    9.16403240e-007,   4.44432790e-009],
 [  1.92126683e-034,   4.48051789e-001,   6.82616225e-116,  3.69215780e-096,
    2.47562962e-022,  1.21289534e-003, 6.33180473e-002,   7.70642917e-010,
    4.87417268e-001,  1.50760551e-081],
 [  9.68958090e-016,   8.85625290e-002,   8.15661099e-003,  3.91213654e-075,
    1.23643521e-001,   3.89004206e-042,   6.57682916e-003,   2.59187469e-018,
    9.93559869e-003,   7.63124912e-001]])
# The means of each componen
means = np.array([ [-4.8148830,  1666.08891501],
                   [-1.38935210e-02,   3.91030687e+03],
                   [-1.99192974,  1046.23895782],
                   [62.74867408,  5900.35504954],
                   [-3.20580073,  2117.37933441],
                   [5.41826534e+00,   5.55629291e+03],
                   [28.32309588,  12085.3822865],
                   [1.42299360e+00,   2.82856492e+03],
                   [2.76268151e+00,   6.98788862e+03],
                   [3.04984218e-01,   1.10614339e+03]])
# The covariance of each component
covars = np.array([[19901.10899247,  145559.1353942 ],
                   [3797.02038489,  1235931.58385831],
                   [13182.17439996,  87137.57994132],
                   [1759475.42864922,  36552747.45908708],
                   [2724.71340548, 296602.83220848],
                   [63837.66522882,  2867629.16600791],
                   [20513.28086561,  19980338.31462503],
                   [28962.97633114,  520482.13848515],
                   [4315.55389006,  3128607.93648248],
                   [1790.20488976,  123237.84834907]])

# Build an HMM instance and set parameters
test_model = GaussianHMM(n_components=10, covariance_type="diag")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
test_model.startprob_ = startprob
test_model.transmat_ = transmat
test_model.means_ = means
test_model.covars_ = covars

test_hidden_states = test_model.predict(X)
print(test_hidden_states)

test_result= test_hidden_states
for i in range(len(test_close_v_real)):
    if test_result[i]==0:
        test_result[i]=-1
    elif test_result[i]==1:
        test_result[i] = -1
    elif test_result[i]==2:
        test_result[i] = -1
    elif test_result[i] ==4:
        test_result[i] = -1
    else:
        test_result[i]=1


print()
print("test_result:")
print(test_result)


for i in range(len(test_close_v_real)):
    if test_close_v_real[i]<0:
        test_close_v_real[i]=-1
    else: test_close_v_real[i] = 1


print()
print("test_real:")
print(test_close_v_real)

compare=[]
compare=test_result-test_close_v_real
#print()
#print("compare:")
#print(compare)
right= compare[compare==0]
#print (right)
#print(len(right))
print(len(right)/len(test_result))

#print(sum(abs(compare)))