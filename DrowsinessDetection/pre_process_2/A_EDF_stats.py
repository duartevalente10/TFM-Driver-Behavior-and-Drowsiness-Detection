import pyedflib
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from datetime import datetime

# read .edf file
def read_edf(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signals = []
    signal_labels = f.getSignalLabels()
    sample_rate = f.getSampleFrequency(3)
    print("Sample Frequency ECG: ", sample_rate)
    num_samples = f.getNSamples()
    print("Numero de samples: ", num_samples)

    duration = num_samples / sample_rate

    # get start time
    start_time_str = f.getStartdatetime().strftime('%Y-%m-%d %H:%M:%S')
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    
    for i in range(n):
        signal = f.readSignal(i)
        signals.append(signal)
    
    f._close()
    return signals, signal_labels,duration, start_time

# plot signals
def plot_signals(signals, signal_labels):
    # num of signals
    num_signals = len(signals)

    # create a figure with subplots
    plt.figure(figsize=(12, 2 * num_signals))

    # plot each signal in a subplot
    for i, signal in enumerate(signals):
        plt.subplot(num_signals, 1, i + 1)
        plt.plot(signal)
        plt.title(f"Signal {i} ({signal_labels[i]})")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")

    # prevent overlap
    plt.tight_layout()
    plt.show()

# file path
#file_path = '../datasets_2/valu3s/vitaport/fp19_1.edf'
file_path = '../datasets_3/DROZY/psg/1-1.edf'

signals, signal_labels, duration, start_time = read_edf(file_path)

print("DURATION:", duration )

print("Start time:", start_time )

# print the signal lengths and labels
for i, (label, signal) in enumerate(zip(signal_labels, signals)):
    print(f"Signal {i} ({label}) length: {len(signal)}")

# basic statistics
signal_stats = []
for signal in signals:
    stats = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal)
    }
    signal_stats.append(stats)

# signal statistics
for i, stats in enumerate(signal_stats):
    print(f"Signal {i} statistics: {stats}")

# plot
plot_signals(signals, signal_labels)