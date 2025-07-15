import pandas as pd
import os
import glob

# func to create a fromated dataset from a folder with all csv files
def formatData(folder_path, name_to_save):
    # get all csv files
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    # list to store the dfs
    dfs = []

    # loop each file
    for file_path in file_paths:
        # extract the file name
        file_name = os.path.basename(file_path)

        print(file_name)
        
        # format the name 
        parts = file_name.split('_')
        prefix = parts[2]  # fpfp11
        sequence_number = parts[3]  # 1
        formatted_prefix = 'fp' + prefix[4:]  # fp + 01
        formatted_name = formatted_prefix + '_' + sequence_number + '.edf' # fp01 + _ + 1 + .edf

        # read the csv
        df = pd.read_csv(file_path, delimiter=';')
        
        # add Filename
        df['Filename'] = formatted_name
        
        # append to the list
        dfs.append(df)

    # merge dfs in the list
    merged_df = pd.concat(dfs, ignore_index=True)

    # filter for 'vitaport_value' == 0
    merged_df = merged_df[merged_df['vitaport_value'] == 0]

    # select features
    merged_df = merged_df[['timer [s]', 'kss_answer', 'Filename']]

    # save to csv
    merged_df.to_csv(f'datasets/sim/{name_to_save}.csv', index=False)

    print(f"Saved as '{name_to_save}.csv'.")

    return merged_df

# Format two sim folders 

# path
folder_path_sim_1 = '../datasets_2/valu3s/sim/simulator_1_elvagar'
folder_path_sim_2 = '../datasets_2/valu3s/sim/simulator_2_blaljus'

sim1 = formatData(folder_path_sim_1,"sim_1_kss_filter_34")
sim2 = formatData(folder_path_sim_2,"sim_2_kss_filter_34")

# Merge two sim datasets
merged_df = pd.concat([sim1, sim2], ignore_index=True)

# filter out rows where kss_answer is -1
filtered_df = merged_df[merged_df['kss_answer'] != -1]

# save new dataframe
filtered_df.to_csv('datasets/sim/sim_filered_34.csv', index=False)

