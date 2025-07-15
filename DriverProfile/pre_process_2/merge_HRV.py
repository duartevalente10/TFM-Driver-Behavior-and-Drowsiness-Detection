import pandas as pd

## 2 Min datasets

# # load dfs
# time = pd.read_csv('hrv_time_domain_2_min.csv')
# freq = pd.read_csv('hrv_freq_domain_2_min.csv')

# # merge dfs
# merged_df = pd.merge(time, freq, on=['Interval_Start', 'Interval_End', 'Filename'])

# # save
# merged_df.to_csv('hrv_time_freq_2_min.csv', index=False)


## 5 Min datasets

# # load dfs
# time = pd.read_csv('hrv_time_domain_5_min.csv')
# freq = pd.read_csv('hrv_freq_domain_5_min.csv')

# # merge dfs
# merged_df = pd.merge(time, freq, on=['Interval_Start', 'Interval_End', 'Filename'])

# # save
# merged_df.to_csv('hrv_time_freq_5_min.csv', index=False)


## 5 Min datasets Filtered

# # load dfs
# time = pd.read_csv('datasets/hrv/hrv_time_domain_5_min_filtered.csv')
# freq = pd.read_csv('datasets/hrv/hrv_freq_domain_5_min_filtered.csv')

# # merge dfs
# merged_df = pd.merge(time, freq, on=['Interval_Start', 'Interval_End', 'Filename'])

# # save
# merged_df.to_csv('hrv_time_freq_5_min_filtered.csv', index=False)

## 2 Min datasets Filtered

# load dfs
time = pd.read_csv('datasets/hrv/hrv_time_domain_2_min_filtered.csv')
freq = pd.read_csv('datasets/hrv/hrv_freq_domain_2_min_filtered.csv')

# merge dfs
merged_df = pd.merge(time, freq, on=['Interval_Start', 'Interval_End', 'Filename'])

# save
merged_df.to_csv('hrv_time_freq_2_min_filtered.csv', index=False)
