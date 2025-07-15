import numpy as np
import pyedflib
import pandas as pd
from collections import defaultdict


# read marker and time data from the .edf
def read_edf_marker(edf_file_path):
    # open .edf 
    f = pyedflib.EdfReader(edf_file_path)
    
    # gett sampling frequency 
    marker_sampling_rate = f.getSampleFrequency(3) 
    
    # get marker signal
    marker_signal = f.readSignal(3)  
    
    # number of samples and compute the time array based on sampling rate
    n_samples = len(marker_signal)
    time = np.arange(n_samples) / marker_sampling_rate 

    f.close()
    return marker_signal, time

# get total duration for each unique value in the signal
def compute_total_duration(signal, time, min_duration=1e-6):
    # get unique values
    unique_values = np.unique(signal)
    # dicionary to store durations
    duration_per_value = defaultdict(float)
    # index for the start of current value
    start_idx = None
    # the fist value of the signal
    current_value = signal[0]

    # loop for each instance of the signal
    for i in range(1, len(signal)):
        # if the current value change 
        if signal[i] != current_value:
            # if is not at the end of the signal, compute the duration of the "previous" value 
            if start_idx is not None:
                end_idx = i - 1
                start_time = time[start_idx]
                end_time = time[end_idx]
                duration = end_time - start_time
                # only add if it is a relevant duration
                if duration >= min_duration: 
                    duration_per_value[current_value] += duration
            # update current value
            current_value = signal[i]
            # set start of new value
            start_idx = i
        elif start_idx is None:
            start_idx = i

    # for the unique value of signal 
    if start_idx is not None:
        end_idx = len(signal) - 1
        start_time = time[start_idx]
        end_time = time[end_idx]
        duration = end_time - start_time
        # only add if it is a relevant duration
        if duration >= min_duration:
            duration_per_value[current_value] += duration

    return duration_per_value

# process vitaport_values
def compute_vitaport_durations(csv_file_path):
    # get data
    df = pd.read_csv(csv_file_path, delimiter=';')
    
    # get vitaport values
    vitaport_signal = df['vitaport_value'].values

    # get sim timer
    time = df['timer [s]'].values
    
    # get durations for each unique value of vitaport_value
    vitaport_durations = compute_total_duration(vitaport_signal, time)
    
    # print total duration for each unique value of vitaport value
    print("Total duration for each unique value of vitaport feature from the Sim Dataset:")
    for value, duration in vitaport_durations.items():
        print(f"Value: {value}, Total Duration: {duration:.2f} seconds")
    
    return vitaport_durations

# process marker from edf
def compute_marker_durations(edf_file_path):
    # get marker and time
    marker_signal, time = read_edf_marker(edf_file_path)
    
    # get total durations for each unique value
    marker_durations = compute_total_duration(marker_signal, time)
    
    # print total duration for each unique value
    print("\nTotal duration for each unique value in Marker from the EDF file:")
    for value, duration in marker_durations.items():
        print(f"Value: {value}, Total Duration: {duration:.2f} seconds")
    
    return marker_durations

# file paths
# edf_file_path = '../datasets_2/valu3s/vitaport/fp09_4.edf' 
# csv_file_path = '../datasets_2/valu3s/sim/simulator_1_elvagar/valu3s_db_fpfp09_4_night_1678498178438.csv' 

edf_file_path = '../datasets_2/valu3s/vitaport/fp19_1.edf' 
csv_file_path = '../datasets_2/valu3s/sim/simulator_1_elvagar/valu3s_db_fpfp19_1_day_1679066465406.csv' 

# edf_file_path = '../datasets_2/valu3s/vitaport/fp01_1.edf' 
# csv_file_path = '../datasets_2/valu3s/sim/simulator_1_elvagar/valu3s_db_fpfp01_1_dag_day_1678117344096.csv' 

# edf_file_path = '../datasets_2/valu3s/vitaport/fp14_3.edf' 
# csv_file_path = '../datasets_2/valu3s/sim/simulator_2_blaljus/valu3s_db_fpfp14_3_day_1678836400420.csv'

# vitaport unique values durations
vitaport_durations = compute_vitaport_durations(csv_file_path)

# marker unique values durations 
marker_durations = compute_marker_durations(edf_file_path)

