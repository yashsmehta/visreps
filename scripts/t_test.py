from scipy.stats import ttest_ind

group1 = [23, 21, 24, 25, 20]
group2 = [27, 29, 26, 30, 28]

t_stat, p_val = ttest_ind(group1, group2)
print(f"T-statistic: {t_stat}, p-value: {p_val}")