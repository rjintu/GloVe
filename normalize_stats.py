import numpy as np
from statistics import NormalDist
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

all_test_stats = []

regions = ['northeast', 'south', 'midwest', 'west', 'federal']
times = ['1860-1889', '1890-1919', '1920-1949', '1950-1979', '1980-now']
# colors = ['red', 'yellow', 'green', 'blue', 'purple']

# real_test_stats = []

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
        #real_test_stats.append(norm_stat)
        print(f'{filename}: {norm_stat}')
        curr.close()

# print(real_test_stats)
# print('***')
# diff_real_test_stats = [x - real_test_stats[-1] for x in real_test_stats][:-1]
# print(diff_real_test_stats)

# randomized
for i in range(20):
    curr = open(f'bias_test_output_{i}.txt', 'r')
    curr.readline()
    test_stat = float(curr.readline().split(' ')[-1])
    mean = float(curr.readline().split(' ')[-1])
    stddev = float(curr.readline().split(' ')[-1])

    # normalize the test statistic
    norm_stat = (test_stat - mean) / (stddev) # 1000 samples
    print(f'bias_test_output_{i}: {norm_stat}')
    all_test_stats.append(norm_stat)
    curr.close()

# calculate all combs of differences
all_diffs = [(i - j) for i, j in combinations(set(all_test_stats), 2)]

all_np = np.array(sorted(all_diffs))

mu, std = np.mean(all_np), np.std(all_np)
curr_dist = NormalDist(mu=mu, sigma=std)
sns.set_style('whitegrid')
plot = sns.kdeplot(all_np, bw=0.5)

# for i in range(len(diff_real_test_stats)):
#     plt.axvline(x=diff_real_test_stats[i], color=colors[i])
#     # plt.axvline(x=critical_val, linestyle='--')

# plot.set_title("Distribution of Subset Distances for Randomized Corpora")
# plot.set_xlabel("Distance between normalized test statistics")
# plt.savefig('randomized_data_diffs_diff.png')
