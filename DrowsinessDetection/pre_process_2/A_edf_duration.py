import os
import pandas as pd
import pyedflib

# path to directory
#edf_directory = '../datasets_2/valu3s/vitaport/'
edf_directory = '../datasets_3/DROZY/psg/'

# get all EDF files
edf_files = [f for f in os.listdir(edf_directory) if f.endswith('.edf')]

# vars to store names and durations
file_names = []
durations = []

# get the duration for each edf file
for edf_file in edf_files:
    edf_path = os.path.join(edf_directory, edf_file)
    with pyedflib.EdfReader(edf_path) as f:
        duration = f.file_duration
        file_names.append(edf_file)
        durations.append(duration)

# create df
df = pd.DataFrame({
    'Filename': file_names,
    'Duration': durations
})

# durations
print(df)

# save
#df.to_csv('datasets/hrv/hrv_duration.csv', index=False)