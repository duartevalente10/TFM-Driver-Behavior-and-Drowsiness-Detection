import os
import pandas as pd

# folder path of filtered csv
folder_path = '../datasets_2/valu3s/vitaport/filtered_signals/' 

# list to store filename and duration
data = []

# loop all files 
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # get file path
        file_path = os.path.join(folder_path, filename)
        
        # read the csv
        df = pd.read_csv(file_path)
        
        # count the number of instances 
        num_instances = len(df)
        
        # get duration ( instaces x sampling frequency of the ECG)
        duration = round(num_instances / 256, 1)

        # change filename 
        new_filename = filename[:-4] + '.edf'  
        
        # append to the list
        data.append({'Filename': new_filename, 'Duration': duration})

# create a df
result_df = pd.DataFrame(data)

# output path
output_file_path = 'datasets/hrv/filtered_ECG_duration.csv' 

# save as csv
result_df.to_csv(output_file_path, index=False)