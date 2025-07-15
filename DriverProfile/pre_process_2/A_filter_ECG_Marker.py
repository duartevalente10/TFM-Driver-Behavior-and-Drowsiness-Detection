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

def ecg_marker_filter(ecg_signal, marker_signal, time_ecg, time_marker):

    # resample marker signal to match ECG time axis using linear interpolation
    resampled_marker_signal = np.interp(time_ecg, time_marker, marker_signal)

    # Create a mask for signal = 34
    mask = resampled_marker_signal == 34

    # filter ECG signal using the mask
    filtered_signal = ecg_signal * mask 

    # remove zeros from the filtered signal
    filtered_signal = filtered_signal[filtered_signal != 0]

    return filtered_signal

# plot signals
def plot_signals(signal, signal_label):

    # create a figure with subplots
    plt.figure(figsize=(12, 2))

    plt.plot(signal)
    plt.title(f"Filtred Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # prevent overlap
    plt.tight_layout()
    plt.show()

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


# edf_file_path = '../datasets_2/valu3s/vitaport/fp19_1.edf' 

# ecg_signal, marker_signal, time_ecg, time_marker, edf_sampling_rate, marker_sampling_rate  = read_edf_marker(edf_file_path)

# filtered_signal = ecg_marker_filter(ecg_signal, marker_signal, time_ecg, time_marker)

# # plot signal
# plot_signals(filtered_signal, "Filtered ECG Signal")


# Usage
input_folder = '../datasets_2/valu3s/vitaport/'  
output_folder = '../datasets_2/valu3s/vitaport/filtered_signals/'  

process_edf_folder(input_folder, output_folder)