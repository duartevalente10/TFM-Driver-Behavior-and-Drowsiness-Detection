import pandas as pd

"""
pre_process_V2.merge_HRV_Sim
----------------------------

This module provides an aproach to merge HRV dataset with the sim dataset

    Datasets:
        - hrv_time_freq_5_min 
            - dataset with time domain and frequency domain features
            - 927 instances of intrevals of 5 min
            - with filenames for association with sim data
            - not labeled

        - hrv_time_freq_2_min 
            - dataset with time domain and frequency domain features
            - 2371 instances of intrevals of 2 min
            - with filenames for association with sim data
            - not labeled

        - sim_data_timer_kss
            - each instance represents an instant ( in seconds) in the simulation
            - each instance is associated to one filename
            - each instance is labeled with KSS value

    Problems:
       - the duration each HRV filename recording is diferent from each sim recording
       - sim data have timestamps but HRV dont
    
    Aproach:
        - group the sim dataset in intrevals of 5 min and 2 min too
        - merge the sim data in midle of the HRV recording by filename
        - merge the sim data in beggining of the HRV recording by filename
        - merge the sim data at the end of the HRV recording by filename

    Funcion: 
        - process_and_merge_intervals
    
"""

def process_and_merge_intervals(data_HRV: pd.DataFrame, data_Sim: pd.DataFrame, interval_duration, start_offset) -> pd.DataFrame:
    """
    Process the HRV and Sim datasets to create and align x min intervals.
    
    Parameters:
        data_HRV: HRV data 
        data_Sim: Sim data
        interval_duration: number of seconds of the intreval
        start_offset: starting offset in seconds
    
    Returns:
        df with HRV and Sim data merged on x min intervals
    """

    # Adjust `timer [s]` to account for the start offset
    data_Sim['Adjusted_Timer'] = data_Sim['timer [s]'] - start_offset
    # remove negative values
    data_Sim['Adjusted_Timer'] = data_Sim['Adjusted_Timer'].clip(lower=0)

    # split data_Sim in x min intervals
    data_Sim['Interval'] = (data_Sim['Adjusted_Timer'] // interval_duration) * interval_duration + start_offset
    
    # group Sim data by Filename and Interval and compute the mean KSS
    Sim_intervals = data_Sim.groupby(['Filename', 'Interval'])['kss_answer'].mean().reset_index()
    Sim_intervals.rename(columns={'Interval': 'Interval_Start'}, inplace=True)
    Sim_intervals['Interval_End'] = Sim_intervals['Interval_Start'] + interval_duration

    # round kss 
    Sim_intervals['kss_answer'] = Sim_intervals['kss_answer'].round()

    # merge
    merged_data = pd.merge(data_HRV, Sim_intervals, on=['Filename', 'Interval_Start', 'Interval_End'], how='inner')

    print(Sim_intervals)
    print(data_HRV)

    # Filter out filenames that are not present in both datasets
    common_filenames = set(data_HRV['Filename']).intersection(set(data_Sim['Filename']))
    filtered_data = merged_data[merged_data['Filename'].isin(common_filenames)]
    
    return filtered_data

### Load Data ###

## 2 min  ##

# load HRV dataset 2 min
# file_path_2_min = 'datasets/hrv/hrv_time_freq_2_min.csv'
# data_HRV = pd.read_csv(file_path_2_min, delimiter=',')

## 5 min  ##

# # # load HRV dataset 5 min
# file_path_HRV= 'datasets/hrv/hrv_time_freq_5_min.csv'
# data_HRV = pd.read_csv(file_path_HRV, delimiter=',')

## 5 min Filtered ##

# # load HRV dataset 5 min filtered
# file_path_HRV_filtered= 'datasets/hrv/hrv_time_freq_5_min_filtered.csv'
# data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')

# ## 2 min Filtered ##

# # # load HRV dataset 2 min filtered
# file_path_HRV_filtered= 'datasets/hrv/hrv_time_freq_2_min_filtered.csv'
# data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')

# ## 1 min Filtered ##

# # # load HRV dataset 1 min filtered
# file_path_HRV_filtered= 'datasets_v2/before_merge/time_freq_1_min.csv'
# data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')

# ## 3 min Filtered ##

# # # load HRV dataset 3 min filtered
# file_path_HRV_filtered= 'datasets_v2/before_merge/time_freq_3_min.csv'
# data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')

## 8 min Filtered ##

# # load HRV dataset 8 min filtered
# file_path_HRV_filtered= 'datasets_v2/before_merge/time_freq_8_min.csv'
# data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')^

## 1 min  Overlap Filtered ##

# # load HRV dataset 1 min overlap filtered
file_path_HRV_filtered= 'datasets_v2/before_merge/time_freq_1_min_overlap.csv'
data_HRV = pd.read_csv(file_path_HRV_filtered, delimiter=',')

## Sim  ##

# load Sim dataset
file_path_Sim = 'datasets/sim/sim_filered_34.csv'
data_Sim = pd.read_csv(file_path_Sim, delimiter=',')

# replace '.edf' with '.csv' 
data_Sim['Filename'] = data_Sim['Filename'].str.replace('.edf', '.csv')

### Merge datasets ###

# process and merge 
final_result = process_and_merge_intervals(data_HRV, data_Sim, 60, 30)

# save
#final_result.to_csv('datasets_v2/supervised_1_min.csv', index=False)
#final_result.to_csv('datasets_v2/supervised_2_min.csv', index=False)
#final_result.to_csv('datasets_v2/supervised_3_min.csv', index=False)
#final_result.to_csv('datasets_v2/supervised_5_min.csv', index=False)
#final_result.to_csv('datasets_v2/supervised_8_min.csv', index=False)
final_result.to_csv('datasets_v2/before_merge/supervised_1_min_overlap.csv', index=False)

