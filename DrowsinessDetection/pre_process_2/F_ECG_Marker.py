import numpy as np
import pyedflib
import pandas as pd
import matplotlib.pyplot as plt
import os

# read marker and time data from the .edf
def read_edf_marker(edf_file_path):
    # open .edf 
    f = pyedflib.EdfReader(edf_file_path)
    
    # get EDF sampling frequency 
    edf_sampling_rate = f.getSampleFrequency(2) 

    # get Marker sampling frequency 
    marker_sampling_rate = f.getSampleFrequency(3) 

    # get ecg signal
    ecg_signal = f.readSignal(2) 
    
    # get marker signal
    marker_signal = f.readSignal(3) 

    # number of samples ECG
    n_samples_ecg = len(ecg_signal)

    # number of samples Marker
    n_samples_marker = len(marker_signal)

    time_ecg = np.arange(n_samples_ecg) / edf_sampling_rate 

    time_marker = np.arange(n_samples_marker) / marker_sampling_rate 

    print("EDF Sampling Rate: ", edf_sampling_rate, "Hz")
    print("Marker Sampling Rate: ", marker_sampling_rate, "Hz")

    print("Number of samples ECG: ", n_samples_ecg)
    print("Number of samples Marker: ", n_samples_marker)

    f.close()
    return ecg_signal, marker_signal, time_ecg, time_marker, edf_sampling_rate, marker_sampling_rate 

# use the marker signal = 34 as a mask on the ECG
def ecg_marker_filter(ecg_signal, marker_signal, time_ecg, time_marker):

    # resample marker signal to match ECG time axis using linear interpolation
    resampled_marker_signal = np.interp(time_ecg, time_marker, marker_signal)

    # create a mask for signal = 34
    mask = resampled_marker_signal == 34

    # filter ECG signal using the mask
    filtered_signal = ecg_signal * mask 

    # remove zeros from the filtered signal
    filtered_signal = filtered_signal[filtered_signal != 0]

    return filtered_signal

# for each edf, filter the ECG with the marker = 34
def process_edf_folder(input_folder, output_folder):
    # create output folder 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # loop for all files 
    for filename in os.listdir(input_folder):
        if filename.endswith('.edf'):
            edf_file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")
            ecg_signal, marker_signal, time_ecg, time_marker, edf_sampling_rate, marker_sampling_rate = read_edf_marker(edf_file_path)

            # filter
            filtered_signal = ecg_marker_filter(ecg_signal, marker_signal, time_ecg, time_marker)

            # save filtred signal
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
            pd.DataFrame(filtered_signal).to_csv(output_file_path, index=False, header=False)
            print(f"Filtered signal saved to: {output_file_path}")

# input and output folder
input_folder = '../datasets_2/valu3s/vitaport/'  
output_folder = '../datasets_2/valu3s/vitaport/filtered_signals/'  

# process all edf files
process_edf_folder(input_folder, output_folder)