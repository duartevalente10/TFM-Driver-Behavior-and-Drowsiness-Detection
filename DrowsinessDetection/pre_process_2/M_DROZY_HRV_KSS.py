import pandas as pd
import numpy as np

# load dataset 
file_path= 'DROZY_datasets/DROZY_time_freq_1_min.csv'
data = pd.read_csv(file_path, delimiter=',')

# get both numbers from the filename column and convert them to integers
data[['part1', 'part2']] = data['Filename'].str.extract(r'(\d+)-(\d+)').astype(int)

# sort the df 
data = data.sort_values(by=['part1', 'part2'])

# prop the temporary columns used for sorting
data = data.drop(columns=['part1', 'part2'])

# unique filenames
unique_filenames = data['Filename'].unique()

# print the filenames
for filename in unique_filenames:
    print(filename)

# KSS values from the txt file
with open('../datasets_3/DROZY/KSS.txt', 'r') as file:
    kss_values = [list(map(int, line.split())) for line in file]

# flatten the KSS values
kss_flattened = np.array(kss_values).flatten()

print(kss_flattened)

# indices to remove 
indices_to_remove = [18,24,28,34,35,38]

# remove indices
kss_flattened = np.delete(kss_flattened, indices_to_remove)

print(kss_flattened)

# repeat each KSS value to match the number of instances per filename
repeats_per_filename = len(data) // len(kss_flattened)  
kss_repeated = np.repeat(kss_flattened, repeats_per_filename)

print(len(kss_repeated))

# add kss to dataset
data['kss_answer'] = kss_repeated

# save supervised dataset
data.to_csv('DROZY_supervised_time_freq_1_min.csv', index=False)

