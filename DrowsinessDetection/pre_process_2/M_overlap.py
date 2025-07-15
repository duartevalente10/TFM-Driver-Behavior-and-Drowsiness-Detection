import pandas as pd

# get original dataset
file_path = 'datasets_v2/supervised_1_min.csv'
data = pd.read_csv(file_path, delimiter=',')

# get offseted dataset
file_path_overlap = 'datasets_v2/before_merge/supervised_1_min_overlap.csv'
data_overlap = pd.read_csv(file_path_overlap, delimiter=',')

print("Dataset starting on 0: ",data.shape)
print("Dataset starting on 30: ",data_overlap.shape)

# concatenate the datasets
combined_data = pd.concat([data, data_overlap])

# sort by filename and start time
combined_data.sort_values(by=['Filename', 'Interval_Start'], inplace=True)

# reset index
combined_data.reset_index(drop=True, inplace=True)

print(combined_data.shape)

combined_data.to_csv('datasets_v2/supervised_1_min_overlap.csv', index=False)