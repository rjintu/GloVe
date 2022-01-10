# use this specifically for averages

import numpy as np
from statistics import NormalDist
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

all_test_stats = []

regions = ['northeast', 'south']
times = ['1860-1889', '1890-1919', '1920-1949', '1950-1979', '1980-now']
# colors = ['red', 'yellow', 'green', 'blue', 'purple']

# real_test_stats = []
south_stats = []
northeast_stats = []

# all regions from more recent
for time in times:
    for region in regions:
        filename = f'output_{region}_{time}.txt'
        curr = open(filename, 'r')
        curr.readline()
        test_stat = float(curr.readline().split(' ')[-1])
        mean = float(curr.readline().split(' ')[-1])
        stddev = float(curr.readline().split(' ')[-1])

        norm_stat = (test_stat) / (stddev) # 1000 samples
        if region == 'south':
            south_stats.append(norm_stat)
        else:
            northeast_stats.append(norm_stat)
        #real_test_stats.append(norm_stat)
        # print(f'{filename}: {norm_stat}')
        curr.close()

diffs = []
for i in range(len(south_stats)):
    diffs.append(northeast_stats[i] - south_stats[i])

mean_diff = np.mean(np.array(diffs))
print(mean_diff)

# randomized
for i in range(20):
    curr = open(f'bias_test_output_{i}.txt', 'r')
    curr.readline()
    test_stat = float(curr.readline().split(' ')[-1])
    mean = float(curr.readline().split(' ')[-1])
    stddev = float(curr.readline().split(' ')[-1])

    # normalize the test statistic
    norm_stat = (test_stat - mean) / (stddev) # 1000 samples
    all_test_stats.append(norm_stat)
    curr.close()

# calculate all combs of differences
all_diffs = [(i - j) for i, j in combinations(set(all_test_stats), 2)]

all_np = np.array(sorted(all_diffs))

critical_val = all_np[-10] # above this value is 5%

mu, std = np.mean(all_np), np.std(all_np)
curr_dist = NormalDist(mu=mu, sigma=std)
sns.set_style('whitegrid')
plot = sns.kdeplot(all_np, bw=0.5).set_title("Mean difference between Northeast and South across time periods")

plt.axvline(x=mean_diff)
plt.axvline(x=critical_val, linestyle='--')
plt.savefig('diff_ne_south.png')

# for i in range(len(diff_real_test_stats)):
#     plt.axvline(x=diff_real_test_stats[i], color=colors[i])
#     # plt.axvline(x=critical_val, linestyle='--')

# plot.set_title("Distribution of Subset Distances for Randomized Corpora")
# plot.set_xlabel("Distance between normalized test statistics")
# plt.savefig('randomized_data_diffs_diff.png')
