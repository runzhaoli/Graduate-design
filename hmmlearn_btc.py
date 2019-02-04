from __future__ import print_function

import datetime
import time

import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

from hmmlearn.hmm import GaussianHMM


print(__doc__)

###############################################################################


def conv_time(t):
    timefmt = '%Y/%m/%d %H:%M'
    return time.mktime(time.strptime(t, timefmt))

dates_list = []
close_v_list = []
close_v_list1 = []
volume_list_diff = []
fn_date = 'date_h.txt'
fn_diff = 'diff_h.txt'
fn_diff1 = 'diff_h1.txt'
fn_vol_diff = 'vol_h_diff.txt'
with open(fn_date) as fp:
    for line in fp:
        dates_list.append(conv_time(line.strip()))


with open(fn_diff) as fp:
    for line in fp:
        try:
            close_v_list.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise

with open(fn_diff1) as fp:
    for line in fp:
        try:
            close_v_list1.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise

with open(fn_vol_diff) as fp:
    for line in fp:
        try:
            volume_list_diff.append(float(line.strip()))
        except:
            print('ERROR line:', line.strip())
            raise

dates = np.array(dates_list, dtype=int)
close_v = np.array(close_v_list)#差价
close_v1 = np.array(close_v_list1)
volume_diff = np.array(volume_list_diff)
X = np.column_stack([close_v1, volume_diff])
###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
print("done")
#print()
hidden_states = model.predict(X)
print(hidden_states)
print()
print()

###############################################################################
print("startprob：")
print(model.startprob_)#初始
print()
print("Transition matrix")
print(model.transmat_)#状态转移概率矩阵
print()
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])#每个状态的平均参数
    print("var = ", np.diag(model.covars_[i]))#每个状态的协方差参数
    print()
'''
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))

    # Format the ticks.
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())

    ax.grid(True)

plt.show()

hidden_states_test=hidden_states;
for i in range(len(hidden_states)):
    if hidden_states_test[i]==4:
        hidden_states_test[i]=-1
    elif hidden_states_test[i]==5:
        hidden_states_test[i] = -1
    elif hidden_states_test[i]==7:
        hidden_states_test[i] = -1
    elif hidden_states_test[i] == 9:
        hidden_states_test[i] = -1
    elif hidden_states_test[i] == 10:
        hidden_states_test[i] = -1
    elif hidden_states_test[i] == 12:
        hidden_states_test[i] = -1
    elif hidden_states_test[i] == 14:
        hidden_states_test[i] = -1
    else:
        hidden_states_test[i]=1


print()
print("hidden_states_test:")
print(hidden_states_test)


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
'''